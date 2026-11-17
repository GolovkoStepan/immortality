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

## Перед тегом или push — `just ci` целиком

**Урок `v0.1.3`.** Тег был поставлен при красном джобе `dependencies`. Локально ничего не
«пропустили»: `just ci` был объявлен как `fmt-check lint test`, то есть **не** запускал ни
`cargo deny`, ни проверку слоёв, хотя комментарий над ним обещал «exactly what GitHub Actions
runs». Комментарий не падает, тест — падает.

Правило: **перед тегом, push в `master` или релизом гоняется `just ci`, а не только `cargo test`.**

```sh
just ci     # fmt-check + lint + test + audit + layering — все четыре джоба
```

Если `just` нет: `brew install just`. `cargo deny` — `cargo install cargo-deny --locked`;
он единственный требует сеть (база RustSec тянется на каждом прогоне, поэтому результат меняется
без коммитов — так и покраснел `v0.1.3`).

Паритет держит `crates/daemon/tests/ci_contract.rs`: он парсит `justfile` и `ci.yml` и падает,
если джоб есть в CI и нет в `ci`, если `audit` сузился, или если запись в `deny.toml` `ignore`
осталась без `D-NNN` в комментарии над ней. Добавляешь джоб в CI — добавь рецепт, строку в
`JOB_COVERAGE` и зависимость `ci`.

`just ci` идёт на macOS и о Linux-джобе молчит. Если push покраснел там, а локально всё зелёное,
прогон воспроизводится контейнером — рецепт в [`docs/linux-check.md`](docs/linux-check.md).
Класс, который так ловится: тест зовёт macOS-only бинарь (`plutil`, `launchctl`) — такие тесты
обязаны быть под `#![cfg(target_os = "macos")]`. Обратный класс — код и тесты, которые зовут
имя, экспортируемое только на Linux или только на macOS: `pub use` под `cfg` означает, что
остальная часть крейта видит разный API на разных платформах, и ошибку показывает только сборка
там. Для сервиса это решено тем, что оба бэкенда (`service::launchd`, `service::systemd`)
публичны всегда, а `pub use` выбирает лишь то, что видит CLI; тесты импортируют модуль по имени.

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
