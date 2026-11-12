# Code rules

How to write and rewrite code. The folder layout is in `02-ARCHITECTURE.md`; rules for
comments and language are in `docs/process/protocol.md` §11. The examples below are in Python, for illustration;
the principles apply to any language. Your stack's own conventions (test layout and naming, commands) are
in `02-ARCHITECTURE.md`, section "Stack conventions".

## The main thing: behavior doesn't change silently

Refactoring changes how the code is built, but not what users, external services and the server see.
**These stay unchanged:**
- texts that users and the client see (messages, page labels), byte for byte;
- log event names and their fields: production searches rely on them;
- HTTP paths, parameters, response codes and response bodies;
- table and column names, values written to the database, idempotency keys;
- environment variable names and their defaults, `*_ENABLED` / `*_DRY_RUN` flags;
- background job ids, their periods and order;
- which exceptions escape the module and which are caught (see "Exceptions").

A bug found along the way **is not fixed** in the refactoring: it is recorded in `07-RISKS.md`
and fixed in a separate commit when the owner says so.

## Clean, quality code

All new and rewritten code follows clean code practices and OOP principles. This is an
acceptance condition: code that works but is hard to read is not done.

**Principles:**
- **Single responsibility (SRP).** A module, class and function each have one reason to change.
  A function does one thing and is named after it; if you want an "and" in the name, it's two functions.
- **Open for extension (OCP).** A new variant (channel, delivery method, event type)
  is added as a new class with a shared interface, not as another `if kind == ...` branch.
- **Substitution (LSP).** A subclass or interface implementation is interchangeable with the base: the same
  arguments, the same meaning of the result, the same exceptions.
- **Narrow interfaces (ISP).** A dependency is described by what the consumer actually needs
  (an interface of a couple of methods), not "the entire external service client".
- **Dependency on abstractions (DIP).** A service receives clients, settings and the session factory
  through its constructor rather than creating them inside. That way it can be tested with stubs and the
  implementation swapped without touching business logic.
- **Encapsulation.** A class's state changes only through its methods; internals are hidden (`_`).
  Process state (caches, in-memory counters) sits behind an object with a meaningful name, not in a
  bare global dictionary.
- **Composition over inheritance.** Inheritance only for a true "is a";
  reuse happens by assembling objects.
- **DRY.** One piece of knowledge in one place: a rule, a key format, a text, a constant.
- **KISS and YAGNI.** The simplest solution that solves the problem now. No "for the future"
  layers and settings nobody uses.

**Practices:**
- Short functions (guideline: up to 30–40 lines), one level of abstraction within a function.
- Early return (`if not x: return`) instead of deep nesting; more than three levels of
  indentation is a reason to split.
- No "magic" numbers and strings in logic: named constants and enumerations.
- Type hints on all public functions and methods; `Any` only at the boundary with foreign JSON.
- Immutable default values; side effects are explicit in the name (`save_…`, `send_…`).
- Errors are not swallowed silently; exceptions are meaningful.
- Dead and commented-out code is not added (only the owner removes existing code of that kind).
- Code reads top to bottom: public stuff above, helpers below or in a separate module.

## Modules and packages

- One module, one responsibility. Guideline: up to ~300 lines; more is a reason to split.
- A large module becomes a package with modules per process step.
- Code is organized into folders by meaning: `core/` (foundation: settings, database, logging, errors),
  `integrations/` (foreign formats and external APIs), `features/<area>/` (business logic),
  `api/` (entry points), `jobs/` (background jobs), `cli/` (commands). Adapt to the project.
- Code constants are enumerations (`StrEnum`): values stay strings, and a typo becomes
  an error at import time.
- Values without behavior are immutable dataclasses instead of dictionaries where the data shape
  is fixed. Dictionaries from external APIs are parsed in one place (a parser) into a typed object.

## Classes

A class is introduced when it simplifies things, not "to have OOP":
- **there are dependencies and state**: a service class with dependencies in the constructor;
  a test passes stubs to the constructor rather than patching global functions;
- **there are variants of one action**: a shared interface and a class per variant;
- **a background job**: a class with a `run_once(now)` method; the scheduler's entry-point function
  stays a thin wrapper: assemble dependencies, call `run_once`.

A pure function without dependencies stays a function.

## Names

- The name makes clear what it does: `parse_incoming_message`, `create_order_for_client`, not `_do`,
  `res`, `d`. Same for files, folders and classes: `incoming_messages.py`, not `inbound.py`;
  `RetryPolicy`, not `RP`.
- A function is a verb (`find_open_order`), a boolean is a question (`is_retryable_error`, `has_open_order`),
  a class is a noun, a collection is plural.
- A leading `_` only for what isn't used outside the module.
- Renaming a public name happens together with all callers and tests in the same commit.

## Exceptions

- Custom exceptions live in a shared errors module or in the area's module, with meaningful names and
  a message saying "what happened and why".
- **Catching behavior doesn't change** during refactoring: a new exception inherits from the one
  that was caught before.
- A broad `except Exception` only at a boundary (background job, webhook handler) and
  always with a log entry; swallowing silently is not allowed.
- Don't lose the cause: `raise NewError(...) from exc`.

## Security

- Secrets (tokens, keys, webhook URLs) are not written to logs or exception texts.
- Personal data (phone, name, conversation text) stays out of logs, or only at DEBUG level.
- Data from webhooks and external APIs is untrusted: type and shape are checked at the boundary, not deep inside.
- Protections (signature checks, locks, idempotency keys, "the DB session isn't held during
  a network call") are not weakened or bypassed when moving code.

## Comments and docstrings

- In the project language (`CLAUDE.md`), explaining "why", not retelling the code. No decision numbers or dates.
- A docstring on every module (what's in it and where its neighbors are), on every class and on public
  functions/methods. On private ones when the "why" isn't obvious.

## Tests change along with code

Test folders mirror code folders **completely, with all subpackages**: tests for the module
`app/<path>/<module>.py` live in `tests/<path>/test_<module>.py` (Python example; your stack's convention is in `02-ARCHITECTURE.md`).

- **File name**: `test_<module>.py`; several files per module get a qualifier:
  `test_<module>_stress.py` (hostile scenarios), `test_<module>_<what>.py`.
- **Changed behavior: update tests in the same commit.** A commit that changes behavior without tests is not done.
- **New module: new test file** in the mirrored test folder.
- **Moved or renamed a module: move its tests.**
- **Fixed a bug: first a test that catches it** (fails before the fix, passes after).
- Before committing, run your area's tests; the full run happens at the end of the block or in CI.

## Tests during refactoring

- Test assertions (`assert`) don't change. Only patch and import targets, names of
  renamed functions, and the way a stub is supplied change.
- If a test can't be kept without changing an assertion, stop and investigate:
  most likely the behavior changed.
- Dangerous modules (acting on real people, money, other people's data) are covered before rewriting
  with snapshot tests (reference output on a set of inputs), validated by mutations.
