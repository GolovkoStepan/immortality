# Инструменты и политика

## 1. Единая точка проверки

Все инструменты проходят через один `PolicyGate`:

```rust
pub enum Effect {
    Read,        // ничего не меняет вне процесса
    Mutate,      // меняет файлы, git, процессы — в пределах worktree
    External,    // выходит за пределы машины или необратимо
}

pub trait Tool {
    fn name(&self) -> &'static str;
    fn schema(&self) -> &'static serde_json::Value;
    fn idempotency(&self, args: &Args) -> Idempotency;          // Idempotent | ByContent | Unknown
    fn effect(&self, args: &Args) -> Effect;                    // может зависеть от аргументов
    fn probe(&self, rec: &CallRecord) -> Probe;                 // Applied | NotApplied | Unknown
    fn execute(&self, ctx: &ToolCtx, args: Args) -> ToolResult;
}
```

`PolicyGate::check(tool, args) -> Allowed | Escalate(reason) | Denied(reason)`:

1. `Effect::Read` → всегда разрешено.
2. `Effect::Mutate` → path allowlist + `deny_write_globs` + правило «только внутри worktree».
3. `Effect::External` → в очередь эскалаций, никогда не исполняется автоматически, даже если
   в конфиге стоит `allow_* = true` (флаг лишь позволяет пометить эскалацию как
   auto-approved для конкретных хостов или типов).
4. Любая попытка выйти за `workspace_root` → `Denied` + событие в журнал (это индикатор
   промпт-инъекции или зацикливания).

`Denied` — это observation модели, а не падение: она получает «действие запрещено
политикой, причина X, разрешённые альтернативы Y».

## 2. Каталог v0.1

### Чтение (Effect::Read)

| Инструмент | Параметры | Примечания |
| --- | --- | --- |
| `read_file` | `path`, `offset`, `limit` | Нумерация строк; отказ на бинарных файлах с подсказкой |
| `glob` | `pattern`, `path` | Сортировка по mtime, лимит результатов |
| `grep` | `pattern`, `glob`, `path`, `-A/-B`, `limit` | Регексы; поиск идёт только по текстовым файлам |
| `read_state` | `section` | Читает `PLAN.md` / `TASKS.json` / `NOTES.md` / `DECISIONS.md` |
| `grep_journal` | `query`, `since_seq`, `limit` | Поиск по журналу задачи; возвращает сводку |
| `git status/diff/log` | параметры git | Только чтение |

### Изменение (Effect::Mutate)

| Инструмент | Параметры | Идемпотентность |
| --- | --- | --- |
| `write_file` | `path`, `content` | ByContent |
| `edit_file` | `path`, `old`, `new`, `replace_all` | ByContent |
| `git add/commit/checkout/branch/worktree/stash` | параметры git | commit → Unknown (probe по nonce) |
| `update_state` | `file`, `content` или `patch` | ByContent |
| `shell` (mutating) | `command`, `timeout_secs`, `cwd` | Unknown |
| `start_process` | `command`, `cwd`, `env`, `name` | Unknown (probe по маркеру в командной строке, D-061) |
| `kill_process` | `handle` | Idempotent |

`edit_file` — точная замена строки, по образцу read-tool'ов: если `old` не найден или
найден неоднократно, возвращается структурированная ошибка (модель должна уточнить
контекст), а не «первое вхождение». Это заметно снижает класс ошибок «исправил не то».

### Внешнее (Effect::External)

| Инструмент | Поведение |
| --- | --- |
| `escalate` | Создать запрос человеку: `key`, `question`, `options`, `blocking: bool` |
| любой сетевой доступ | Только через эскалацию; хосты из `trusted_hosts` разрешены без вопроса (сюда входит `llm-server`) |
| `delegate` | Суб-агент с чистым контекстом в том же worktree: `brief`, `scope`, `max_steps` |

### Завершение

| Инструмент | Поведение |
| --- | --- |
| `finish_subtask` | Заявляет выполнение DoD; гейты проверяются **системой**, а не моделью |
| `finish_task` | Заявляет выполнение задачи; запускаются все гейты, включая reviewer |

## 3. Terminal

- **Persistent shell на задачу**: cwd/env сохраняются между вызовами, состояние живёт в
  демоне, не в модели.
- **stdin закрыт по умолчанию.** Команда, которая ждёт ввода, получает EOF и не вешает цикл.
- **Таймаут на вызов** (дефолт 120 с, максимум задаётся конфигом). При истечении процесс
  убивается (SIGTERM → SIGKILL через 5 с), возвращается структурированный результат
  `timeout` с тем, что успело выйти.
- **Автоперевод в фон**: если вывода нет N секунд или команда помечена как долгоживущая,
  она становится tracked-процессом с `handle`, а модели возвращается
  `{"handle": "proc_7", "hint": "read_process"}`.
- **Ограничение вывода**: всё пишется в artifact, в контекст — последние 200 строк +
  сводка (код выхода, число строк, найденные ошибки/ссылки на файлы:строка).
- **Классификатор**: `read` (git status, ls, cat, grep, cargo test/check, npm run lint…),
  `mutate` (mkdir, mv, sed -i, cargo build, npm install в worktree…),
  `external` (curl, wget, ssh, scp, git push/fetch, brew, sudo, любые обращения вне localhost).
  Незнакомая команда консервативно считается `external`.

## 4. Эскалации

Очередь, которую агент видит как pending. Форма:

```json
{
  "id": "esc_01JB...",
  "key": "network:example.com:fetch-docs",
  "task": "tsk_01JB...",
  "kind": "external_action" | "blocked" | "backend_unavailable" | "journal_corrupt" | "disk",
  "question": "Разрешить загрузку …?",
  "options": ["allow_once", "allow_for_task", "deny"],
  "blocking": false,
  "created_at": "...",
  "resolved_at": null,
  "answer": null
}
```

Правила:

- `blocking: false` (по умолчанию) — агент продолжает работу по другим подзадачам.
- Дедуп по `key`: одинаковый вопрос не задаётся дважды.
- Ответ человека — событие в журнал; `allow_for_task` живёт до конца задачи, не дольше.
- Эскалация — не конец прогона. Человек может ответить из `immortality escalations`
  в любой момент, даже посреди работы.
- macOS-нотификация (`osascript`) на открытие эскалации, без внешних каналов в v0.1.
- `blocking: true` используется только когда без ответа физически нельзя продолжать
  ни одну подзадачу (например, `journal_corrupt`).

## 5. Гейты DoD

Завершение проверяет система по `ACCEPTANCE.md`:

1. **Команды проверки** (сборка, тесты, линт) — исполняются, результат в журнал
   (`gate_result`). Красный гейт = подзадача не закрыта.
2. **Reviewer-субагент** с чистым контекстом: получает только diff и DoD, возвращает
   вердикт `approve` / `request_changes` со списком проблем. Не может править код сам.
3. **Дифф-санити**: нет случайных бинарников, нет секретов (эвристика по паттернам
   `API_KEY=`, приватные ключи), нет изменений в `.git/config`, нет правок глобального
   конфига, нет `TODO(agent)` без причины.

Только при трёх зелёных гейтах подзадача → `done`, а задача при всех закрытых подзадачах →
`ready_for_review`. Модель не может объявить успех в обход проверок.
