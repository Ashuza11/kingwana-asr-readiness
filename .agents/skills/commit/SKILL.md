---
name: commit
description: Stage or commit explicitly requested Git changes after a privacy review, select and show one conventional commit message, and push only when explicitly requested. Use when the user invokes $commit or asks to stage, commit, or push changes in this research repository.
---

# Commit requested changes

1. Run `git status` and inspect the relevant unstaged and staged diffs.
2. Stage files only when the user explicitly requests staging. Never broaden the requested scope.
3. Before committing, inspect staged paths and content for participant audio, personal information,
   consent records, `.env`, credentials, model weights, or unapproved research data. Warn and stop
   if anything is suspicious.
4. Summarize the staged changes and why they exist.
5. Choose exactly one conventional commit type:
   - ✨ `feat:`
   - 🐛 `fix:`
   - 🔨 `refactor:`
   - 📝 `docs:`
   - 🎨 `style:`
   - ✅ `test:`
   - ⚡ `perf:`
6. Choose one concise present-tense commit message and show it to the user before running the
   commit.
7. If the user explicitly requested a commit, commit in the same turn after the checks pass. If the
   user asked only for a proposal, do not commit.
8. Push only when the user explicitly requests it. Report the remote and branch that were pushed.
