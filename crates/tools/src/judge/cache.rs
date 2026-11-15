//! The verdict cache: the same command shape is judged once, not once per call.
//!
//! # Why a cache and not a call per command
//!
//! D-111 measured the shape of the problem: the cost of the judge is per *unknown program*, not
//! per command. A model that runs `bundle exec rspec spec/a_spec.rb` and then
//! `bundle exec rspec spec/b_spec.rb` has asked the same question twice — the program is the same
//! unknown `bundle`, and whether it may run does not depend on which spec was named. Without a
//! cache the judge would be paid per file, which turns a one-off question into a cost proportional
//! to the size of the test suite.
//!
//! # What the key is, and why it is not the command
//!
//! The key is the **signature**: the program plus the *shape* of its arguments, with values
//! replaced by placeholders. `rspec spec/a_spec.rb` and `rspec spec/b_spec.rb` share a signature,
//! while `rspec --format json` does not — a flag changes what the command does, and an argument
//! that is a path does not. That is the distinction the judge's answer actually depends on.
//!
//! The signature includes the policy fingerprint (`yolo`, the deny lists, the trusted hosts).
//! Without it, turning `allow_network` off would leave every cached `allow` for a network command
//! still standing, and the cache would outvote the operator's own configuration. With it, an old
//! verdict simply stops being found and the command is judged again.
//!
//! # Why the cache stores `Allow` only
//!
//! A cached `deny` and a cached `escalate` are both answers a changed world could make wrong, and
//! both cost the model a real step to discover. `allow` is the only verdict a cache is *for*: it is
//! the permission that would otherwise be asked again, and re-asking it is the cost the cache
//! exists to remove. So `deny` and `escalate` are not stored — the next identical command asks
//! again, which is cheap and cannot be stale in the dangerous direction.
//!
//! # Why it survives `kill -9`
//!
//! The verdicts are written to the journal as [`immortality_core::journal::Kind::JudgeVerdict`]
//! records, so a daemon that dies mid-run does not re-pay for every unknown program at the next
//! start. The journal is single-writer and the gate does not hold it — see
//! [`crate::judge::Recorded`] for how the two are kept apart.

use std::collections::BTreeMap;

use super::{Judgement, key};
use crate::classify::External;

/// One judgement, as it is remembered and as it is written to the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// The command shape this answers, from [`signature`].
    pub signature: String,
    /// The policy fingerprint this answer was given under, from [`policy_fingerprint`].
    pub policy: String,
    /// The command as the model wrote it, for the journal and for a human reading it back.
    pub command: String,
    /// The answer.
    pub judgement: Judgement,
}

/// The program and the shape of its arguments, with values removed.
///
/// `bundle exec rspec spec/a_spec.rb` becomes `bundle exec rspec <path>`, so a suite of a hundred
/// files costs one judgement rather than a hundred. A flag keeps its own name, because a flag
/// changes what the command does: `rspec --format json` is a different question from
/// `rspec spec/a_spec.rb`.
///
/// Never fails. An empty command has an empty signature, which is a key nothing else shares and a
/// command the shell refuses anyway.
#[must_use]
pub fn signature(command: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for word in command.split_whitespace() {
        // A flag is a fact about the command; a value is not. `--format=json` keeps the flag and
        // drops the value, `spec/a_spec.rb` becomes a placeholder.
        if let Some(flag) = word.strip_prefix("--") {
            let name = flag.split('=').next().unwrap_or(flag);
            out.push(format!("--{name}"));
            continue;
        }
        if word.starts_with('-') && word.len() > 1 {
            // A short flag may be clustered (`-xvf`) or take a value (`-n5`). Only the letters are
            // part of the shape; a digit is a value.
            let letters: String = word
                .chars()
                .skip(1)
                .take_while(char::is_ascii_alphabetic)
                .collect();
            if letters.is_empty() {
                out.push(word.to_string());
            } else {
                out.push(format!("-{letters}"));
            }
            continue;
        }
        if looks_like_value(word) {
            out.push("<arg>".to_string());
        } else {
            out.push(word.to_string());
        }
    }
    out.join(" ")
}

/// Whether a word is a value rather than part of the command's shape.
///
/// Deliberately conservative: a path, a number and a word containing `=` are values, and everything
/// else — a subcommand like `exec` or `run` — is part of the shape. Erring towards "this is
/// structure" costs an extra judgement; erring the other way would let `npm run build` and
/// `npm run deploy` share one verdict, which is the direction that matters.
fn looks_like_value(word: &str) -> bool {
    word.contains('/')
        || word.contains('.')
        || word.contains('=')
        || word.chars().all(|ch| ch.is_ascii_digit())
        || word.starts_with('~')
        || word.starts_with('$')
}

/// A fingerprint of everything in the policy that changes what a judge answer means.
///
/// Not the whole policy: the worktree, the glob lists for reading and the commit/install flags do
/// not change whether an unknown program is safe to run. What does is `yolo` (which waives the
/// question), the program deny list, the write deny list and the trusted hosts.
#[must_use]
pub fn policy_fingerprint(policy: &immortality_core::config::Policy) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("yolo={}", policy.yolo));
    let mut deny = policy.deny_programs.clone();
    deny.sort();
    parts.push(format!("deny_programs={}", deny.join(",")));
    let mut write = policy.deny_write_globs.clone();
    write.sort();
    parts.push(format!("deny_write_globs={}", write.join(",")));
    let mut hosts: Vec<String> = policy.trusted_hosts.iter().cloned().collect();
    hosts.sort();
    parts.push(format!("trusted_hosts={}", hosts.join(",")));
    // A digest rather than the joined text: the journal stores this on every record, and the
    // lists are unbounded in principle.
    let joined = parts.join(";");
    immortality_core::journal::digest_text(&immortality_core::journal::hash_line(joined.as_bytes()))
}

/// The judgements remembered so far, by key.
///
/// `Mutex` and not `RwLock` because it is written on every miss and read on every hit, and the
/// critical section is a map lookup: the contention that would justify `RwLock` is not there.
#[derive(Debug, Default)]
pub struct Cache {
    hits: std::sync::Mutex<BTreeMap<String, Verdict>>,
    /// Judgements made during this process, in order, for the caller to journal.
    ///
    /// Separate from `hits` because the two have different lifetimes: `hits` is the working set and
    /// shrinks to nothing between runs, while this is what has not been written to the journal yet
    /// and must survive exactly until it is.
    fresh: std::sync::Mutex<Vec<Verdict>>,
}

impl Cache {
    /// An empty cache. Seeds itself from `known`, which the caller reads back from the journal.
    #[must_use]
    pub fn new(known: impl IntoIterator<Item = Verdict>) -> Self {
        let cache = Self::default();
        if let Ok(mut hits) = cache.hits.lock() {
            for verdict in known {
                // Only `allow` is remembered: see the module docs. A journal written by an older
                // build may hold a `deny` or an `escalate`, and seeding those would make the cache
                // answer questions it was explicitly not built to answer.
                if matches!(verdict.judgement, Judgement::Allow) {
                    hits.insert(verdict.key(), verdict);
                }
            }
        }
        cache
    }

    /// The remembered answer for this key, if there is one.
    #[must_use]
    pub fn get(&self, signature: &str, policy: &str) -> Option<Judgement> {
        let key = key(signature, policy);
        self.hits
            .lock()
            .ok()
            .and_then(|hits| hits.get(&key).map(|verdict| verdict.judgement.clone()))
    }

    /// Remember an answer.
    ///
    /// Only `allow` is stored — see the module docs — but every answer is returned by
    /// [`Self::take_fresh`] so that the journal records what the judge actually said.
    pub fn insert(&self, verdict: Verdict) {
        if matches!(verdict.judgement, Judgement::Allow)
            && let Ok(mut hits) = self.hits.lock()
        {
            hits.insert(verdict.key(), verdict.clone());
        }
        if let Ok(mut fresh) = self.fresh.lock() {
            fresh.push(verdict);
        }
    }

    /// The judgements not yet written to the journal, clearing them.
    ///
    /// Called by whoever holds the journal. The gate cannot: `run_tool` holds `&mut Journal` for
    /// the whole call, so a gate that wrote there would need a second writer on a single-writer
    /// file.
    #[must_use]
    pub fn take_fresh(&self) -> Vec<Verdict> {
        self.fresh
            .lock()
            .map(|mut fresh| std::mem::take(&mut *fresh))
            .unwrap_or_default()
    }

    /// How many answers are remembered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hits.lock().map_or(0, |hits| hits.len())
    }

    /// Whether nothing is remembered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Verdict {
    /// The map key, from the signature and the policy fingerprint.
    #[must_use]
    pub fn key(&self) -> String {
        key(&self.signature, &self.policy)
    }

    /// The same verdict as the payload of a
    /// [`Kind::JudgeVerdict`](immortality_core::journal::Kind::JudgeVerdict) record.
    ///
    /// Flat, like every other record's payload: the fields are read with `payload["verdict"]` by
    /// a human and by [`Self::from_payload`], and a nested object would make the one query a
    /// post-mortem asks — "what did the judge allow?" — a path expression.
    #[must_use]
    pub fn to_payload(&self) -> serde_json::Value {
        let mut body = serde_json::Map::new();
        body.insert(
            "command".to_string(),
            serde_json::Value::String(self.command.clone()),
        );
        body.insert(
            "signature".to_string(),
            serde_json::Value::String(self.signature.clone()),
        );
        body.insert(
            "policy".to_string(),
            serde_json::Value::String(self.policy.clone()),
        );
        body.insert(
            "verdict".to_string(),
            serde_json::Value::String(self.judgement.as_str().to_string()),
        );
        if let Some(why) = self.judgement.why() {
            body.insert("why".to_string(), serde_json::Value::String(why.to_string()));
        }
        serde_json::Value::Object(body)
    }

    /// The verdict a journal payload describes, or `None` when it has been damaged.
    ///
    /// `None` rather than an error: a record that cannot be read back is one cached permission
    /// that is not found, and re-judging the command is cheap and safe. Refusing the whole task
    /// over it would trade a model call for a dead end.
    #[must_use]
    pub fn from_payload(payload: &serde_json::Value) -> Option<Self> {
        let text = |field: &str| payload.get(field).and_then(serde_json::Value::as_str);
        let why = || text("why").unwrap_or_default().to_string();
        let judgement = match text("verdict")? {
            "allow" => Judgement::Allow,
            "deny" => Judgement::Deny(why()),
            "escalate" => Judgement::Escalate(why()),
            _ => return None,
        };
        Some(Self {
            signature: text("signature")?.to_string(),
            policy: text("policy")?.to_string(),
            command: text("command")?.to_string(),
            judgement,
        })
    }
}

/// Whether a command's structural reason is one a judge may answer at all.
///
/// A free function rather than a call to `External::is_a_question` at each site, so the one place
/// that decides the boundary is named and tested.
#[must_use]
pub const fn may_judge(source: Option<External>) -> bool {
    matches!(source, Some(source) if source.is_a_question())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(command: &str, judgement: Judgement) -> Verdict {
        Verdict {
            signature: signature(command),
            policy: "policy-a".to_string(),
            command: command.to_string(),
            judgement,
        }
    }

    #[test]
    fn the_signature_keeps_the_program_and_the_shape_and_drops_values() {
        // The cost D-111 names is per unknown program, not per command: a suite of a hundred files
        // must cost one judgement, and the measured case is exactly that — `bundle exec rspec`
        // followed by a different spec file.
        assert_eq!(
            signature("bundle exec rspec spec/a_spec.rb"),
            signature("bundle exec rspec spec/b_spec.rb")
        );
        assert_eq!(
            signature("bundle exec rspec spec/a_spec.rb"),
            "bundle exec rspec <arg>"
        );
        // A flag is part of the shape: it changes what the command does.
        assert_ne!(
            signature("rspec --format json"),
            signature("rspec --format progress")
        );
        // `--flag=value` drops the value, because the two halves are one word and the flag is the
        // part that matters.
        assert_eq!(signature("rspec --format=json"), "rspec --format");
        // The space-separated form keeps the bare word, which is the deliberate conservative
        // direction: it costs an extra judgement (`--format json` and `--format progress` are two
        // questions) and never lets one verdict answer for a different command.
        assert_eq!(signature("rspec --format json"), "rspec --format json");
    }

    #[test]
    fn a_subcommand_is_structure_and_a_path_is_a_value() {
        // The line the cache must not cross: `npm run build` and `npm run deploy` are different
        // questions, while two files passed to the same program are not. Erring towards "value"
        // here would let one verdict answer for both.
        assert_ne!(signature("npm run build"), signature("npm run deploy"));
        assert_eq!(signature("npm run build"), "npm run build");
        assert_eq!(signature("node -e 1"), "node -e <arg>");
        assert_ne!(signature("git status"), signature("git push"));
    }

    #[test]
    fn the_policy_fingerprint_changes_with_the_flags_that_matter() {
        use immortality_core::config::Policy;
        let base = Policy::default();
        let fingerprint = policy_fingerprint(&base);
        assert_eq!(
            fingerprint,
            policy_fingerprint(&base),
            "the same policy must fingerprint the same"
        );
        let yolo = Policy {
            yolo: true,
            ..Policy::default()
        };
        assert_ne!(fingerprint, policy_fingerprint(&yolo));
        let deny = Policy {
            deny_programs: vec!["curl".to_string()],
            ..Policy::default()
        };
        assert_ne!(
            fingerprint,
            policy_fingerprint(&deny),
            "a deny entry changes the answer"
        );
        // An empty fingerprint change must not happen for something the judge's answer ignores:
        // `allow_commit` does not make an unknown program safer to run.
        let commit = Policy {
            allow_commit: true,
            ..Policy::default()
        };
        assert_eq!(
            fingerprint,
            policy_fingerprint(&commit),
            "a flag the judge's answer does not depend on must not invalidate it"
        );
    }

    #[test]
    fn only_an_allow_is_remembered() {
        // A cached `deny` or `escalate` would be an answer a changed world could make wrong, and
        // finding out would cost a real step. `allow` is the permission the cache exists to avoid
        // re-asking.
        let cache = Cache::default();
        cache.insert(verdict("bundle exec rspec a.rb", Judgement::Allow));
        cache.insert(verdict("curl x", Judgement::Deny("network".into())));
        cache.insert(verdict("frobnicate", Judgement::Escalate("unclear".into())));
        assert_eq!(cache.len(), 1);
        assert_eq!(
            cache.get(&signature("bundle exec rspec b.rb"), "policy-a"),
            Some(Judgement::Allow),
            "the allowed shape answers a sibling command"
        );
        assert_eq!(cache.get(&signature("curl x"), "policy-a"), None);
        assert_eq!(cache.get(&signature("frobnicate"), "policy-a"), None);
    }

    #[test]
    fn a_different_policy_does_not_find_the_old_allow() {
        // The direction that matters: turning `allow_network` off must not leave a cached `allow`
        // for a network command standing. The fingerprint is part of the key, so the old verdict
        // simply stops being found.
        let cache = Cache::default();
        cache.insert(verdict("bundle exec rspec a.rb", Judgement::Allow));
        assert!(
            cache
                .get(&signature("bundle exec rspec a.rb"), "policy-b")
                .is_none()
        );
    }

    #[test]
    fn every_judgement_is_reported_once_for_the_journal() {
        // The gate cannot write: `run_tool` holds `&mut Journal`. So the cache collects and the
        // caller drains, and a verdict must not be handed out twice or it would be journalled
        // twice.
        let cache = Cache::default();
        cache.insert(verdict("bundle exec rspec a.rb", Judgement::Allow));
        cache.insert(verdict("curl x", Judgement::Deny("network".into())));
        let fresh = cache.take_fresh();
        assert_eq!(
            fresh.len(),
            2,
            "a denial is still reported, even though it is not cached"
        );
        assert!(
            cache.take_fresh().is_empty(),
            "the same verdict must not be reported twice"
        );
    }

    #[test]
    fn a_seeded_cache_answers_without_asking() {
        // What a restart reads back from the journal: the judgement was paid for before the crash,
        // and paying again would change `restarts` for a reason unrelated to the work.
        let known = vec![verdict("bundle exec rspec a.rb", Judgement::Allow)];
        let cache = Cache::new(known);
        assert_eq!(
            cache.get(&signature("bundle exec rspec z.rb"), "policy-a"),
            Some(Judgement::Allow)
        );
        assert!(cache.take_fresh().is_empty(), "a seeded verdict is not new");
    }

    #[test]
    fn a_verdict_survives_the_round_trip_through_a_journal_payload() {
        // The payload is the only channel between the run that paid for a judgement and the run
        // that follows it. A field renamed on one side would make every restart re-ask, and the
        // failure is silent — a miss is not an error.
        for judgement in [
            Judgement::Allow,
            Judgement::Deny("it writes the system gem directory".into()),
            Judgement::Escalate("the command is too broad to answer".into()),
        ] {
            let verdict = verdict("bundle exec rspec a.rb", judgement.clone());
            let payload = verdict.to_payload();
            assert_eq!(Verdict::from_payload(&payload), Some(verdict.clone()));
            assert_eq!(
                payload["command"].as_str(),
                Some("bundle exec rspec a.rb"),
                "a human reading the journal has to see the command, not only its shape"
            );
            assert_eq!(payload["verdict"].as_str(), Some(judgement.as_str()));
        }
        // `allow` carries no `why`: an absent reason is not an empty one.
        let allowed = verdict("x", Judgement::Allow).to_payload();
        assert!(allowed.get("why").is_none(), "{allowed}");
        assert_eq!(
            Verdict::from_payload(&allowed).expect("parse").judgement,
            Judgement::Allow
        );
    }

    #[test]
    fn a_damaged_payload_is_not_found_rather_than_fatal() {
        // A record written by a build that spelled a field differently, or a hand-edited journal,
        // must cost one extra judgement and not the task.
        for payload in [
            serde_json::json!({}),
            serde_json::json!({"verdict": "allow"}),
            serde_json::json!({"verdict": "maybe", "signature": "s", "policy": "p", "command": "c"}),
        ] {
            assert_eq!(Verdict::from_payload(&payload), None, "{payload}");
        }
    }

    #[test]
    fn only_an_unknown_program_may_be_judged() {
        assert!(may_judge(Some(External::UnknownProgram)));
        for source in [
            External::NamedProgram,
            External::UnknownSubcommand,
            External::RemoteSubcommand,
            External::OpaqueScript,
            External::DynamicProgram,
            External::Unreadable,
        ] {
            assert!(!may_judge(Some(source)), "{source:?} is not a question");
        }
        assert!(!may_judge(None), "an External with no source is not routable");
    }
}
