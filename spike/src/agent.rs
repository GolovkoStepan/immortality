//! The measured loop. Deliberately unguarded: no detectors, no budgets, no checkpointing —
//! those are M4. Here we only *observe* and count, because the thresholds for M4 must come
//! from data rather than guesses.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::llm::{AssistantTurn, ChatRequest, Client, Message, ToolCall};
use crate::scenarios::Scenario;
use crate::tools::{self, Sandbox};

#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub model: String,
    pub max_steps: u32,
    pub max_tokens_per_request: u32,
    pub verbose: bool,
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct StepRecord {
    pub step: u32,
    pub elapsed_ms: u128,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cache_hits: u32,
    pub finish_reason: String,
    pub content_chars: usize,
    pub reasoning_chars: usize,
    pub tool_names: Vec<String>,
    pub args_valid: bool,
    pub args_parse_error: Option<String>,
    /// Whether the tool reported failure. A non-zero exit from `check.sh` is expected, so
    /// this is not the same as "the model did something wrong".
    pub outcome_ok: bool,
    pub progress: bool,
    pub duplicate_of_previous: bool,
    pub duplicate_count: u32,
    pub thinking_truncated: bool,
    pub empty_response: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct RunReport {
    pub scenario: String,
    pub model: String,
    pub completed: bool,
    pub stop_reason: String,
    pub steps: u32,
    pub wall_ms: u128,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Sum of prefix-cache hits. The ratio to prompt tokens is the backend health metric:
    /// low cache utilisation means the prompt prefix is changing and every step pays full prefill.
    pub cache_hits: u32,
    pub invalid_tool_calls: u32,
    pub invalid_args: u32,
    pub unknown_tools: u32,
    pub repeated_calls: u32,
    pub max_consecutive_duplicates: u32,
    pub no_progress_steps: u32,
    pub thinking_truncated: u32,
    pub empty_responses: u32,
    pub text_instead_of_tool: u32,
    pub files_touched: usize,
    pub records: Vec<StepRecord>,
}

/// Normalised call identity, used to count exact repeats.
fn call_key(name: &str, args: &Value) -> String {
    let mut normalised = args.clone();
    normalise_json(&mut normalised);
    format!("{name}:{}", normalised)
}

/// Sort object keys and drop volatile fields so that reordered arguments still compare equal.
fn normalise_json(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = std::mem::take(map).into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (key, mut val) in entries {
                normalise_json(&mut val);
                map.insert(key, val);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(normalise_json),
        _ => {}
    }
}

/// Cheap content fingerprint of the sandbox, for detecting steps that changed nothing.
fn tree_hash(root: &Path) -> u64 {
    use std::collections::BTreeMap;
    use std::hash::{Hash, Hasher};

    let mut files: BTreeMap<String, u64> = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(bytes) = std::fs::read(&path) {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                bytes.hash(&mut hasher);
                let rel = path.strip_prefix(root).unwrap_or(&path).display().to_string();
                files.insert(rel, hasher.finish());
            }
        }
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    files.hash(&mut hasher);
    hasher.finish()
}

pub async fn run(
    client: &Client,
    sandbox: &Sandbox,
    scenario: &Scenario,
    config: LoopConfig,
    traces_dir: &Path,
) -> Result<RunReport> {
    (scenario.setup)(sandbox)?;

    let mut report = RunReport {
        scenario: scenario.name.to_string(),
        model: config.model.clone(),
        ..Default::default()
    };
    let mut messages = vec![Message::system(scenario.system), Message::user(scenario.goal)];
    let tool_defs: Vec<Value> = tools::definitions();
    let tool_defs: Vec<crate::llm::Tool> = tool_defs
        .into_iter()
        .map(|value| serde_json::from_value(value).expect("tool definition is valid"))
        .collect();

    let trace_path = traces_dir.join(format!("{}.jsonl", scenario.name));
    let mut trace = std::fs::File::create(&trace_path)?;
    use std::io::Write as _;

    let started = Instant::now();
    let mut last_hash = tree_hash(sandbox.root());
    let mut last_key: Option<String> = None;
    let mut consecutive_duplicates: u32 = 0;

    for step in 1..=config.max_steps {
        let mut request = ChatRequest::new(config.model.clone())
            .max_tokens(config.max_tokens_per_request)
            .tools(tool_defs.clone());
        request.messages = messages.clone();

        let step_started = Instant::now();
        let turn: AssistantTurn = match client.stream_turn(&request, config.verbose).await {
            Ok(turn) => turn,
            Err(err) => {
                report.stop_reason = format!("backend error: {err}");
                writeln!(trace, "{}", serde_json::json!({"step": step, "fatal": err.to_string()}))?;
                break;
            }
        };
        let elapsed_ms = step_started.elapsed().as_millis();

        let mut record = StepRecord {
            step,
            elapsed_ms,
            prompt_tokens: turn.usage.map(|u| u.prompt_tokens).unwrap_or(0),
            completion_tokens: turn.usage.map(|u| u.completion_tokens).unwrap_or(0),
            cache_hits: turn.usage.map(|u| u.cache_n).unwrap_or(0),
            finish_reason: turn.finish_reason.clone(),
            content_chars: turn.content.chars().count(),
            reasoning_chars: turn.reasoning.chars().count(),
            thinking_truncated: turn.is_thinking_truncated(),
            empty_response: turn.content.trim().is_empty() && turn.tool_calls.is_empty(),
            ..Default::default()
        };
        report.prompt_tokens += record.prompt_tokens;
        report.completion_tokens += record.completion_tokens;
        report.cache_hits += record.cache_hits;

        writeln!(
            trace,
            "{}",
            serde_json::json!({
                "step": step,
                "req": {"prompt_tokens": record.prompt_tokens, "max_tokens": config.max_tokens_per_request},
                "resp": {
                    "finish_reason": turn.finish_reason,
                    "content": turn.content,
                    "reasoning_chars": record.reasoning_chars,
                    "tool_calls": turn.tool_calls,
                },
                "elapsed_ms": elapsed_ms,
            })
        )?;

        if config.verbose {
            println!(
                "\n── step {step}  {elapsed_ms}ms  pt={} ct={}  finish={}",
                record.prompt_tokens, record.completion_tokens, turn.finish_reason
            );
            if !turn.reasoning.is_empty() {
                println!("   ~thinking~ {}", snippet(&turn.reasoning, 200));
            }
            if !turn.content.is_empty() {
                println!("   text: {}", snippet(&turn.content, 300));
            }
        }

        if turn.is_thinking_truncated() {
            report.thinking_truncated += 1;
        }
        if record.empty_response {
            report.empty_responses += 1;
        }

        // No tool calls: either the model claims it is done, or it is stalling.
        if !turn.is_tool_call() {
            if (scenario.done)(sandbox) {
                report.completed = true;
                report.stop_reason = "goal met (checked after text answer)".into();
                report.records.push(record);
                break;
            }
            report.text_instead_of_tool += 1;
            messages.push(Message { role: "assistant".into(), content: Some(turn.content.clone()), tool_calls: None, tool_call_id: None, name: None });
            messages.push(Message::user(
                "You replied without calling a tool, but the task is not finished. \
                 Keep working: use the tools. Do not ask me for permission.",
            ));
            report.records.push(record);
            continue;
        }

        record.tool_names = turn.tool_calls.iter().map(|c| c.function.name.clone()).collect();
        // Repeats: an identical call right after the same call is the cheapest hang signal.
        let first_key = call_key(&turn.tool_calls[0].function.name, &first_args(&turn.tool_calls[0]));
        if last_key.as_deref() == Some(first_key.as_str()) {
            consecutive_duplicates += 1;
            record.duplicate_of_previous = true;
            report.repeated_calls += 1;
            report.max_consecutive_duplicates = report.max_consecutive_duplicates.max(consecutive_duplicates);
        } else {
            consecutive_duplicates = 0;
        }
        last_key = Some(first_key);

        messages.push(Message::assistant_tool_calls(turn.tool_calls.clone()));

        let mut all_args_valid = true;
        for call in &turn.tool_calls {
            let parsed = match call.function.parse() {
                Ok(value) => value,
                Err(err) => {
                    all_args_valid = false;
                    record.args_parse_error = Some(err.to_string());
                    report.invalid_args += 1;
                    messages.push(Message::tool_result(
                        &call.id,
                        &call.function.name,
                        format!("ERROR: could not parse arguments as JSON: {err}. Raw: {}", call.function.arguments),
                    ));
                    continue;
                }
            };

            let known = tool_defs.iter().any(|t| t.function.name == call.function.name);
            if !known {
                report.unknown_tools += 1;
                messages.push(Message::tool_result(
                    &call.id,
                    &call.function.name,
                    format!(
                        "ERROR: unknown tool `{}`. Available: read_file, write_file, edit_file, shell.",
                        call.function.name
                    ),
                ));
                continue;
            }

            let outcome = tools::execute(sandbox, &call.function.name, &parsed).await;
            if !outcome.ok {
                report.invalid_tool_calls += 1;
            }
            record.outcome_ok = outcome.ok;
            if config.verbose {
                println!(
                    "   tool: {}({})  -> {}",
                    call.function.name,
                    snippet(&parsed.to_string(), 120),
                    if outcome.ok { "ok" } else { "error" }
                );
                if !outcome.ok {
                    println!("      {}", snippet(&outcome.content, 200));
                }
            }
            messages.push(Message::tool_result(&call.id, &call.function.name, outcome.content));
        }
        record.args_valid = all_args_valid;

        let hash = tree_hash(sandbox.root());
        record.progress = hash != last_hash;
        if !record.progress {
            report.no_progress_steps += 1;
        }
        last_hash = hash;

        report.records.push(record);
        report.steps = step;

        if (scenario.done)(sandbox) {
            report.completed = true;
            report.stop_reason = "goal met".into();
            break;
        }
    }

    report.wall_ms = started.elapsed().as_millis();
    report.files_touched = count_files(sandbox.root());
    if !report.completed && report.stop_reason.is_empty() {
        report.stop_reason = "max steps reached".into();
    }
    Ok(report)
}

fn first_args(call: &ToolCall) -> Value {
    call.function.parse().unwrap_or(Value::Null)
}

fn snippet(text: &str, max: usize) -> String {
    let flat = text.replace('\n', "⏎");
    if flat.chars().count() <= max {
        flat
    } else {
        format!("{}…", flat.chars().take(max).collect::<String>())
    }
}

fn count_files(root: &Path) -> usize {
    let mut count = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                count += 1;
            }
        }
    }
    count
}
