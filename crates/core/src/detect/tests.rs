//! Tests for D1–D3, D6 and the ladder.

use super::*;
use std::path::PathBuf;

fn thresholds() -> Thresholds {
    Thresholds::from_config(&Detectors::default())
}

fn record(signals: &mut Signals, tool: &str, arguments: &str, output: &str) {
    signals.record_call(tool, arguments, Some(output), None);
}

#[test]
fn detector_names_are_stable_because_they_are_a_metric_key() {
    assert_eq!(Detector::ExactRepeat.as_str(), "d1");
    assert_eq!(Detector::SemanticRepeat.as_str(), "d2");
    assert_eq!(Detector::NoProgress.as_str(), "d3");
    assert_eq!(Detector::ReasoningLoop.as_str(), "d4");
    assert_eq!(Detector::Budget.as_str(), "d5");
    assert_eq!(Detector::FileChurn.as_str(), "d6");
}

#[test]
fn no_two_detectors_share_a_name() {
    // The name is the key `core::metrics` tallies by, so two detectors answering to `d1` would be
    // one line in the report and one number for two different loops — a metric that undercounts
    // exactly the thing it exists to count. `as_str` matches exhaustively, so a new variant cannot
    // be added without this list failing to compile; this assertion is what stops it being added
    // under an existing name.
    let names: Vec<&str> = [
        Detector::ExactRepeat,
        Detector::SemanticRepeat,
        Detector::NoProgress,
        Detector::ReasoningLoop,
        Detector::Budget,
        Detector::FileChurn,
    ]
    .into_iter()
    .map(Detector::as_str)
    .collect();
    let mut unique = names.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), names.len(), "two detectors share a name: {names:?}");
}

#[test]
fn no_two_levels_share_a_name() {
    // Same reasoning as the detectors: `d1/l1` and `d1/l2` are different actions taken, and a
    // shared name would merge "nudged" with "restarted the attempt".
    let names: Vec<&str> = [Level::Nudge, Level::ForceStrategy, Level::Restart]
        .into_iter()
        .map(Level::as_str)
        .collect();
    let mut unique = names.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), names.len(), "two levels share a name: {names:?}");
}

#[test]
fn level_names_are_stable() {
    assert_eq!(Level::Nudge.as_str(), "l1");
    assert_eq!(Level::ForceStrategy.as_str(), "l2");
    assert_eq!(Level::Restart.as_str(), "l3");
}

#[test]
fn the_default_thresholds_match_the_documented_ladder() {
    // docs/design.md §7: 3 → L1, 5 → L2, 8 → L3 for D1; 15 → L2, 25 → L3 for D3.
    let thresholds = thresholds();
    assert_eq!(thresholds.exact(2), None);
    assert_eq!(thresholds.exact(3), Some(Level::Nudge));
    assert_eq!(thresholds.exact(4), Some(Level::Nudge));
    assert_eq!(thresholds.exact(5), Some(Level::ForceStrategy));
    assert_eq!(thresholds.exact(7), Some(Level::ForceStrategy));
    assert_eq!(thresholds.exact(8), Some(Level::Restart));
    assert_eq!(thresholds.exact(100), Some(Level::Restart));

    assert_eq!(thresholds.semantic(1), None);
    assert_eq!(thresholds.semantic(2), Some(Level::Nudge));
    assert_eq!(thresholds.semantic(3), Some(Level::ForceStrategy));
    // D2 has no stop: an identical output from two commands is a hint, not proof.
    assert_eq!(thresholds.semantic(50), Some(Level::ForceStrategy));

    assert_eq!(thresholds.progress(14), None);
    assert_eq!(thresholds.progress(15), Some(Level::ForceStrategy));
    assert_eq!(thresholds.progress(24), Some(Level::ForceStrategy));
    assert_eq!(thresholds.progress(25), Some(Level::Restart));
}

#[test]
fn an_identical_call_repeated_three_times_is_a_nudge() {
    let mut signals = Signals::default();
    let thresholds = thresholds();
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "one");
    assert_eq!(signals.react(&thresholds), None);
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "two");
    assert_eq!(signals.react(&thresholds), None);
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "three");
    let reaction = signals.react(&thresholds).expect("the third repeat trips");
    assert_eq!(reaction.detector, Detector::ExactRepeat);
    assert_eq!(reaction.level, Level::Nudge);
    assert_eq!(reaction.metric, 3);
}

#[test]
fn a_different_call_resets_the_exact_repeat_run() {
    let mut signals = Signals::default();
    let thresholds = thresholds();
    for _ in 0..3 {
        record(&mut signals, "read_file", r#"{"path":"a"}"#, "same");
    }
    assert!(signals.react(&thresholds).is_some());
    record(&mut signals, "read_file", r#"{"path":"b"}"#, "other");
    assert_eq!(signals.repeat_run(), 1);
    let mut fresh = Signals::default();
    record(&mut fresh, "read_file", r#"{"path":"a"}"#, "same");
    record(&mut fresh, "read_file", r#"{"path":"b"}"#, "other");
    assert_eq!(signals.repeat_run(), fresh.repeat_run());
}

#[test]
fn the_same_call_with_reordered_keys_counts_as_a_repeat() {
    // A model that shuffles JSON keys is repeating itself; the detector compares canonical
    // forms, not bytes.
    let mut signals = Signals::default();
    record(&mut signals, "shell", r#"{"a":1,"b":2}"#, "out");
    record(&mut signals, "shell", r#"{"b":2,"a":1}"#, "out");
    assert_eq!(signals.repeat_run(), 2);
}

#[test]
fn a_call_with_absolute_and_relative_paths_counts_as_a_repeat() {
    let mut signals = Signals::default();
    let root = PathBuf::from("/work/tree");
    signals.record_call("read_file", r#"{"path":"src/a.rs"}"#, Some("x"), Some(&root));
    signals.record_call(
        "read_file",
        r#"{"path":"/work/tree/src/a.rs"}"#,
        Some("x"),
        Some(&root),
    );
    assert_eq!(signals.repeat_run(), 2);
}

#[test]
fn two_different_commands_with_identical_output_are_a_semantic_repeat() {
    // D2's whole point: the model changes the command, not the experiment.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    record(&mut signals, "shell", r#"{"command":"cat a"}"#, "same text");
    record(&mut signals, "shell", r#"{"command":"cat ./a"}"#, "same text");
    let reaction = signals.react(&thresholds).expect("two identical outputs");
    assert_eq!(reaction.detector, Detector::SemanticRepeat);
    assert_eq!(reaction.level, Level::Nudge);
    record(&mut signals, "shell", r#"{"command":"head a"}"#, "same text");
    let reaction = signals.react(&thresholds).expect("three is L2");
    assert_eq!(reaction.level, Level::ForceStrategy);
}

#[test]
fn the_same_call_twice_is_reported_as_d1_and_not_also_as_d2() {
    // One stuck step is one problem. Without this the third identical call would report D1 and
    // D2 on the same step and the metrics would count two failures where one happened.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    for _ in 0..3 {
        record(&mut signals, "read_file", r#"{"path":"a"}"#, "same");
    }
    // The run of identical outputs stays at one: the repeated call was not fed to D2 at all.
    assert_eq!(signals.semantic_run(), 1);
    let reaction = signals.react(&thresholds).expect("a reaction");
    assert_eq!(reaction.detector, Detector::ExactRepeat);
}

#[test]
fn identical_output_after_real_work_is_not_a_repeat() {
    // Two identical outputs separated by a step that changed something are not a loop: the
    // model tried an experiment in between.
    let mut signals = Signals::default();
    record(&mut signals, "shell", r#"{"command":"a"}"#, "same");
    record(&mut signals, "write_file", r#"{"path":"x"}"#, "wrote x");
    record(&mut signals, "shell", r#"{"command":"b"}"#, "same");
    assert_eq!(signals.semantic_run(), 1);
}

#[test]
fn a_step_that_did_not_run_is_neither_progress_nor_a_repeat() {
    // A refused call or a bad-argument reply is information the model is already being given.
    let mut signals = Signals::default();
    record(&mut signals, "shell", r#"{"command":"a"}"#, "out");
    signals.record_call("shell", "{not json", None, None);
    // It neither fed D2 nor reset D1's run: it is a step that produced no evidence.
    assert_eq!(signals.semantic_run(), 1, "the first call started a run of one");
    assert_eq!(signals.repeat_run(), 1);
}

#[test]
fn a_nudge_is_not_repeated_on_every_subsequent_step() {
    // The run continues after L1, and the model already has the message. Re-sending it spends
    // the context the detector was meant to save.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    for _ in 0..3 {
        record(&mut signals, "read_file", "{}", "same");
    }
    assert_eq!(
        signals.react(&thresholds).map(|reaction| reaction.level),
        Some(Level::Nudge)
    );
    record(&mut signals, "read_file", "{}", "same");
    assert_eq!(signals.react(&thresholds), None, "L1 is already announced");
    record(&mut signals, "read_file", "{}", "same");
    let reaction = signals.react(&thresholds).expect("the fifth repeat trips L2");
    assert_eq!(reaction.level, Level::ForceStrategy);
    assert_eq!(reaction.metric, 5);
}

#[test]
fn fifteen_unproductive_steps_force_a_strategy() {
    let mut signals = Signals::default();
    let thresholds = thresholds();
    for _ in 0..14 {
        signals.record_idle_step();
        assert_eq!(signals.react(&thresholds), None);
    }
    signals.record_idle_step();
    let reaction = signals.react(&thresholds).expect("fifteenth is L2");
    assert_eq!(reaction.detector, Detector::NoProgress);
    assert_eq!(reaction.level, Level::ForceStrategy);
    assert_eq!(reaction.metric, 15);
}

#[test]
fn progress_resets_the_unproductive_stretch() {
    let mut signals = Signals::default();
    let thresholds = thresholds();
    for _ in 0..14 {
        signals.record_idle_step();
    }
    signals.record_progress();
    assert_eq!(signals.unproductive(), 0);
    for _ in 0..14 {
        signals.record_idle_step();
        assert_eq!(signals.react(&thresholds), None);
    }
}

#[test]
fn an_alternation_of_two_known_outputs_is_a_stall_not_a_fresh_step() {
    // The measured failure from the second `black_jack` run: `verify_shuffle_spec` spent 62 steps
    // alternating two commands — `srand(1234); … shuffle!` printing `[41, 28, …]` and an override
    // of `shuffle!` printing `[1, 2, …]` — 28 and 25 times respectively, in a clean A-B-A-B.
    // Both outputs were known from the first two steps, so nothing after them was progress.
    //
    // Every earlier detector was blind by construction: D1 and D2 only count a run of *adjacent*
    // identical calls or outputs, and this sequence has neither. Measured over both live runs, the
    // longest such stretch was 49 in that attempt and 0–6 everywhere else, so a D3 that asks this
    // question separates the two without a new threshold.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    record(
        &mut signals,
        "shell",
        r#"{"command":"a"}"#,
        "[41, 28, 40, 45, 7, 47]",
    );
    record(&mut signals, "shell", r#"{"command":"b"}"#, "[1, 2, 3, 4, 5, 6]");
    assert_eq!(
        signals.react(&thresholds),
        None,
        "two steps, both new — not a stall yet"
    );
    let mut fired = Vec::new();
    for step in 0..30 {
        // Strict A-B-A-B: each step differs from the one before it.
        let (command, output) = if step % 2 == 0 {
            (r#"{"command":"a"}"#, "[41, 28, 40, 45, 7, 47]")
        } else {
            (r#"{"command":"b"}"#, "[1, 2, 3, 4, 5, 6]")
        };
        record(&mut signals, "shell", command, output);
        if let Some(reaction) = signals.react(&thresholds) {
            fired.push(reaction);
        }
    }
    assert_eq!(
        fired.iter().map(|one| one.level).collect::<Vec<_>>(),
        vec![Level::ForceStrategy, Level::Restart],
        "the alternation must reach L2 at 15 and L3 at 25: {fired:?}"
    );
    assert_eq!(fired.first().map(|one| one.detector), Some(Detector::NoProgress));
    assert_eq!(fired.first().map(|one| one.metric), Some(15));
}

#[test]
fn a_known_output_after_real_work_still_stops_being_progress() {
    // The other side of the same rule: re-reading a file the attempt already read teaches nothing,
    // even when the step before it was new. D2 stays silent — these are not adjacent — and this is
    // the signal that catches a model walking a list it has already walked.
    let mut signals = Signals::default();
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "contents of a");
    assert_eq!(signals.unproductive(), 0);
    record(&mut signals, "read_file", r#"{"path":"b"}"#, "contents of b");
    assert_eq!(signals.unproductive(), 0);
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "contents of a");
    assert_eq!(
        signals.unproductive(),
        1,
        "the third step produced nothing this attempt had not seen"
    );
    assert_eq!(signals.semantic_run(), 1, "D2 needs adjacency and must not fire");
}

#[test]
fn a_fresh_output_counts_as_progress() {
    let mut signals = Signals::default();
    for _ in 0..14 {
        signals.record_idle_step();
    }
    assert_eq!(signals.unproductive(), 14);
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "a new fact");
    assert_eq!(signals.unproductive(), 0);
}

#[test]
fn twenty_five_unproductive_steps_ask_for_a_restart() {
    let mut signals = Signals::default();
    let thresholds = thresholds();
    let mut last = None;
    for _ in 0..25 {
        signals.record_idle_step();
        if let Some(reaction) = signals.react(&thresholds) {
            last = Some(reaction);
        }
    }
    let reaction = last.expect("a reaction by the twenty-fifth step");
    assert_eq!(reaction.detector, Detector::NoProgress);
    assert_eq!(reaction.level, Level::Restart);
    assert_eq!(reaction.metric, 25);
}

#[test]
fn an_exact_repeat_is_reported_before_a_semantic_one_on_the_same_step() {
    // One record per step: three records describing one step would inflate the metrics.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    record(&mut signals, "shell", r#"{"command":"a"}"#, "same");
    record(&mut signals, "shell", r#"{"command":"a"}"#, "same");
    record(&mut signals, "shell", r#"{"command":"a"}"#, "same");
    let reaction = signals.react(&thresholds).expect("a reaction");
    assert_eq!(reaction.detector, Detector::ExactRepeat);
}

#[test]
fn the_counters_are_readable_for_the_report() {
    let mut signals = Signals::default();
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "one");
    record(&mut signals, "read_file", r#"{"path":"a"}"#, "two");
    assert_eq!(signals.calls(), 2);
    assert_eq!(signals.repeat_run(), 2);
}

#[test]
fn a_restart_level_has_no_instruction_because_it_ends_the_attempt() {
    assert_eq!(Level::Restart.instruction(Detector::ExactRepeat), None);
    assert_eq!(Level::Restart.instruction(Detector::NoProgress), None);
}

#[test]
fn the_nudge_names_what_repeated_and_asks_for_a_different_step() {
    let text = Level::Nudge
        .instruction(Detector::ExactRepeat)
        .expect("a nudge speaks");
    assert!(text.contains("repeating"), "{text}");
    assert!(text.contains("different"), "{text}");
}

#[test]
fn the_forced_strategy_step_is_the_same_whichever_detector_fired() {
    // The instruction is about how to think, so a D1 loop and a D3 stall produce the same text;
    // a per-detector variant would be four texts nobody proof-reads.
    let text = Level::ForceStrategy
        .instruction(Detector::ExactRepeat)
        .expect("L2 speaks");
    assert!(text.contains("three"), "{text}");
    assert!(text.contains("hypotheses"), "{text}");
    assert_eq!(
        text,
        Level::ForceStrategy.instruction(Detector::NoProgress).unwrap()
    );
}

#[test]
fn a_zero_threshold_never_escalates_that_rung() {
    // `stop: 0` is how D2 says "never restart on this signal alone".
    let ladder = Ladder {
        warn: 1,
        strategy: 2,
        stop: 0,
    };
    assert_eq!(ladder.level(1), Some(Level::Nudge));
    assert_eq!(ladder.level(2), Some(Level::ForceStrategy));
    assert_eq!(ladder.level(1_000), Some(Level::ForceStrategy));
}

#[test]
fn a_count_that_leaps_several_rungs_reports_the_harshest() {
    let ladder = Ladder {
        warn: 3,
        strategy: 5,
        stop: 8,
    };
    assert_eq!(ladder.level(9), Some(Level::Restart));
}

// ── D6: the same file rewritten over and over ────────────────────────────────────

#[test]
fn rewriting_one_file_six_times_is_a_nudge() {
    // The measured failure, reduced to its signal: nineteen successive writes to
    // `spec/probe_spec.rb`, each with a different body, each followed by the same failing check.
    // D1 cannot see it (the bodies differ), D2 and D3 reset on each new output (the check prints
    // random permutations). What stays constant is the destination.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    for _ in 0..5 {
        signals.record_write("write_file", "spec/probe_spec.rb");
        assert_eq!(
            signals.react(&thresholds),
            None,
            "five is below the nudge threshold"
        );
    }
    signals.record_write("write_file", "spec/probe_spec.rb");
    let reaction = signals.react(&thresholds).expect("the sixth write is L1");
    assert_eq!(reaction.detector, Detector::FileChurn);
    assert_eq!(reaction.level, Level::Nudge);
    assert_eq!(reaction.metric, 6);
}

#[test]
fn the_churn_ladder_reaches_force_strategy_and_restart() {
    // The measured run reached nineteen, so the ladder has to have a rung past the nudge —
    // otherwise a model that ignores it spends the whole step budget, which is what happened.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    let mut seen = Vec::new();
    for _ in 0..16 {
        signals.record_write("write_file", "spec/probe_spec.rb");
        if let Some(reaction) = signals.react(&thresholds) {
            seen.push(reaction.level);
        }
    }
    assert_eq!(
        seen,
        vec![Level::Nudge, Level::ForceStrategy, Level::Restart],
        "the ladder must climb, and announce each rung once"
    );
}

#[test]
fn writing_a_different_file_restarts_the_churn_run() {
    // D6 counts one file, not "editing". A model that touches three files in rotation is doing
    // work; a model that rewrites one and then another has moved on.
    let mut signals = Signals::default();
    for _ in 0..5 {
        signals.record_write("write_file", "src/a.rs");
    }
    assert_eq!(signals.churn_run(), 5, "{}", signals.churn_run());
    signals.record_write("write_file", "src/b.rs");
    assert_eq!(signals.churn_run(), 1);
    assert_eq!(signals.churn_path(), Some("src/b.rs"));
}

#[test]
fn the_measured_failure_fires_d6_and_not_d1() {
    // The interleaving from the journal: write the probe, run it, write it again. The run is not
    // reset by the check — that is the whole point, because the check is what makes it a loop.
    //
    // D3 fires later, and correctly so: once the write and the check are both known, every step
    // after them is a repeat. D6 is the one that has to speak *first*, because it names the thing
    // the model can act on ("you are rewriting the same file"), and D1 must stay silent: every
    // body was a different guess.
    let mut signals = Signals::default();
    let thresholds = thresholds();
    let mut fired = Vec::new();
    for _ in 0..7 {
        record(
            &mut signals,
            "write_file",
            r#"{"path":"spec/probe_spec.rb"}"#,
            "Created",
        );
        signals.record_write("write_file", "spec/probe_spec.rb");
        record(
            &mut signals,
            "shell",
            r#"{"command":"bundle exec rspec"}"#,
            "1 example, 1 failure",
        );
        if let Some(reaction) = signals.react(&thresholds) {
            fired.push(reaction.detector);
        }
    }
    assert_eq!(
        fired.first(),
        Some(&Detector::FileChurn),
        "D6 must be the first to speak: {fired:?}"
    );
    assert!(!fired.contains(&Detector::ExactRepeat), "{fired:?}");
}

#[test]
fn progress_ends_a_churn_cycle() {
    // A rewrite after a check that *passed* is the next piece of work, not the same guess again.
    let mut signals = Signals::default();
    for _ in 0..3 {
        signals.record_write("write_file", "src/a.rs");
    }
    signals.record_progress();
    assert_eq!(signals.churn_run(), 0);
    signals.record_write("write_file", "src/a.rs");
    assert_eq!(signals.churn_run(), 1);
}

#[test]
fn a_shell_write_is_not_file_churn() {
    // `Signals::record_write` ignores anything but the two editing tools, so a run of
    // `cargo test > /tmp/out` cannot nudge a model that is working.
    let mut signals = Signals::default();
    for _ in 0..20 {
        signals.record_write("shell", "/tmp/out");
    }
    assert_eq!(signals.churn_run(), 0);
}

#[test]
fn an_absolute_path_and_its_relative_form_are_the_same_file() {
    // Otherwise a model that switches between `spec/a.rb` and `/work/tree/spec/a.rb` would reset
    // its own churn run and evade the detector.
    let root = PathBuf::from("/work/tree");
    let relative = normalize_path("spec/a.rb", &root);
    let absolute = normalize_path("/work/tree/spec/a.rb", &root);
    assert_eq!(relative, absolute, "{relative} vs {absolute}");
}
