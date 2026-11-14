# Правила работы в этом репозитории

Адресовано агенту, который здесь работает. Человек читает [`README.md`](README.md) и
[`docs/`](docs/). Здесь — только то, что агент обязан знать до первого коммита.

## Что можно и что нельзя

**Агент не выполняет `git push` и не настраивает remote.** Работа доводится до коммита в
ветке задачи и на этом останавливается. Решение зафиксировано в
[`docs/non-goals-v0.1.md`](docs/non-goals-v0.1.md) и
[`docs/tools-and-policy.md`](docs/tools-and-policy.md); отсутствие remote — нормальное
состояние, а не поломка.

- **Делает сам:** коммиты в `agent/<task>` ([D-011](docs/decisions.md)), Conventional Commits,
  документацию в том же коммите, что и код.
- **Не делает без человека:** `git push`, `git remote`, публикация пакетов, релизы, merge в
  `main`, правка политик безопасности в глобальном конфиге, `sudo`, сеть вне allowlist,
  удаление вне worktree.

## Проверки перед коммитом

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --offline
cargo test --workspace --offline
```

Линты — **ошибки**, а не предупреждения: `Cargo.toml` держит `warnings = "deny"` и
`unsafe_code = "forbid"` (`[workspace.lints.rust]`), плюс группы `pedantic` и `nursery` и
точечные `unwrap_used`, `expect_used`, `panic`, `todo`, `dbg_macro` (`[workspace.lints.clippy]`).
CI добавляет `clippy -D warnings`. Послабления `unwrap_used`/`expect_used`/`panic`/
`indexing_slicing` объявлены в `lib.rs`/`main.rs` крейтов условно (`cfg_attr(test, allow(...))`),
а интеграционные тесты — безусловным `#![allow(...)]` в начале файла.

Правило M1: **ни один шаг не считается сделанным без теста с инъекцией падения.** Рядом —
саботаж: сломать защиту, убедиться, что тест падает по существу (а не из-за ошибки
компиляции), восстановить.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) — четыре джоба: `fmt`,
`clippy -D warnings` и `test --all-features` на Linux (без `--offline`), `cargo deny check
advisories bans licenses sources`, и проверка слоёв: `immortality-core` не должен обзавестись
`tokio`, `reqwest`, `hyper`, `immortality-tools` или крейтом демона. Воркфлоу триггерится на
**любой** push в любую ветку (и на каждый PR), а не только на `main`. macOS-джоб идёт на каждый
push, а в PR — только с меткой `macos` (это дорогой раннер).

`just ci` = `fmt-check` + `lint` + `test`; `cargo deny` и проверку слоёв он **не** запускает —
зависимости отдельно проверяет `just audit`, а условие по слоям приходится повторять руками
(`grep -E '^(tokio|reqwest|hyper|immortality-daemon|immortality-tools)' crates/core/Cargo.toml`).

## Куда смотреть

| Вопрос | Документ |
| --- | --- |
| Как устроена система, что где лежит | [`docs/design.md`](docs/design.md) |
| Как переживается крах, что в журнале | [`docs/state-and-recovery.md`](docs/state-and-recovery.md) |
| Какой инструмент что делает и что разрешено | [`docs/tools-and-policy.md`](docs/tools-and-policy.md) |
| Почему сделано именно так | [`docs/decisions.md`](docs/decisions.md) |
| Что входит в v0.1 и когда это готово | [`docs/roadmap.md`](docs/roadmap.md), [`docs/non-goals-v0.1.md`](docs/non-goals-v0.1.md) |
| Чем подтверждается качество | [`docs/metrics-v0.1.md`](docs/metrics-v0.1.md) |
| Тестовый репозиторий и прогоны | [`dogfood/README.md`](dogfood/README.md) |
