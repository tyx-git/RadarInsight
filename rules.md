# Project Guidelines

Merged from Karpathy coding guidelines and project-specific rules. Bias toward caution over speed; for trivial tasks use judgment.

## 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs. Before implementing: state assumptions explicitly; if uncertain ask. If multiple interpretations exist, present them. If a simpler approach exists, say so. If something is unclear, stop and ask.

## 2. Simplicity First
Minimum code that solves the problem. Nothing speculative. No features beyond what was asked. No abstractions for single-use code. No "flexibility" or "configurability" not requested. No error handling for impossible scenarios. If 200 lines could be 50, rewrite. Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes
Touch only what you must. Clean up only your own mess. When editing: don't "improve" adjacent code/comments/formatting. Don't refactor unbroken things. Match existing style. Mention unrelated dead code but don't delete it. Remove imports/vars/functions that YOUR changes made unused. Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution
Define success criteria. Loop until verified. Transform tasks into verifiable goals: "Add validation" → write tests for invalid inputs, then make them pass. "Fix bug" → write a reproducing test, then make it pass. For multi-step tasks, state a brief plan with verification steps. Strong criteria let you loop independently.

## 5. Encoding & Chinese Text
Use Python script, not PowerShell, for changes involving Chinese text to avoid mojibake. Prefer existing Python utilities; if none, add a small UTF-8-safe helper. Do not rely on PowerShell commands that emit or rewrite Chinese text.

## 6. Unicode & Mojibake Recovery
If `???` or broken Unicode appears, repair with Python tooling. Prefer reusable repair scripts. Validate affected paths. Keep useful repair scripts in the repo.

## 7. Version Control
Major architectural changes (restructuring, core rewrites, dependency upgrades, migrations, interface changes) must be documented and committed with conventional format: `type(scope): summary`. Update `arch.md` when architecture changes materially.

## 8. Documentation Maintenance
Keep `arch.md` aligned with current architecture and product direction.

## 9. String Language Rules
Code strings in English except comments, docstrings, or human-facing log messages. Use English for identifiers, constants, error messages, API literals, fixed UI text. Improves portability and reduces encoding failures.

## 10. Log Formatting
Use `timestamp [LEVEL] [module] message` with `YYYY-MM-DD HH:MM:SS`, uppercase level, lowercase module with underscores. Example: `2026-03-27 10:44:32 [INFO] [decision] Decision engine ready`

## 11. ASCII Safety
No emoji or decorative Unicode symbols in code, docs, logs, comments. Replace with plain text or ASCII punctuation. Prefer automated Python cleanup.

---

**Success metric:** fewer unnecessary diff changes, fewer overcomplication rewrites, clarifying questions before implementation, rare encoding/logging issues.