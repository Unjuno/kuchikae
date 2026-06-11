# UX Policy

## Principles
1. **Speed is the UX.** The app must feel responsive at every step.
2. **No tutorials.** The UI must be self-explanatory through its layout alone.
3. **Low friction.** One tap to record, one tap to retry. No confirmation dialogs.
4. **No text-heavy UI.** Labels are short (TEMPLATE). Status is 3-4 chars (STT/TXT/VOX/DONE). No long Markdown or instruction copy.
5. **Minimal text.** No verbose helper text, long prompts, onboarding copy, or explanatory blocks.
6. **Prompt iteration.** The prompt textbox is front and center. Users edit and re-record to iterate.
7. **Same-audio prompt iteration must be optimized.** Transcript appears before final voice output where feasible.
8. **Cache repeated work.** Avoid redundant STT and VoiceContext extraction for same audio.
9. **Immediate feedback.** Progressive status shows exactly which stage is running. Audio autoplays on completion.
10. **Lazy-load heavy backends in future real-model integrations.** Initial scaffold focuses on dummy backends for speed.

## Status vocabulary
| Label | Meaning |
|-------|---------|
| STT | Speech-to-text (recognizing) |
| TXT | Text transformation (LLM) |
| VOX | Voice synthesis (TTS) |
| DONE | Complete, audio ready |

## Rejected patterns
- Onboarding walkthroughs or tooltips
- Explanatory text blocks
- Dropdown help menus
- Modal dialogs for non-error states
- Log output in the UI
