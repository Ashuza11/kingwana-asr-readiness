---
name: commit
description: Review already-staged Git changes and propose a privacy-safe conventional commit. Use when the user invokes $commit or asks to prepare a commit from the current index in this research repository.
---

# Commit staged changes

1. Run `git status` and `git diff --staged`.
2. If nothing is staged, report that fact and stop. Never stage files implicitly.
3. Inspect staged paths and content for participant audio, personal information, consent records,
   `.env`, credentials, model weights, or unapproved research data. Warn and stop if anything is
   suspicious.
4. Summarize the staged changes and why they exist.
5. Choose exactly one conventional commit type:
   - ✨ `feat:`
   - 🐛 `fix:`
   - 🔨 `refactor:`
   - 📝 `docs:`
   - 🎨 `style:`
   - ✅ `test:`
   - ⚡ `perf:`
6. Propose one concise, present-tense commit message.
7. Ask for explicit confirmation. Do not commit in the same turn as the proposal.
8. Only after the user approves in a later turn, run the commit with the approved message.
9. Never push automatically.
