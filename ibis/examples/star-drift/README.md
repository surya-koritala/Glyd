# Star Drift, written in one prompt

![Star Drift playing itself](../../docs/media/star-drift.gif)

A space shooter in one HTML file (1,047 lines, no libraries), written by Qwen3.8-Flash-Next on Glyd in **one prompt**, on an
RTX 4080 SUPER 16 GB: precise mode, the default settings (temperature 0.5, min-p 0.1), thinking off. It wrote 16,416 tokens in 98 seconds.
[index.html](index.html) is its reply, unedited.

Open `index.html` in a browser to play (mouse or arrow keys / WASD, Space or click to fire, P to pause), or `index.html#autoplay` to
watch it play itself. It has waves of three enemy types, power-ups, a combo multiplier, a best score, particle explosions, screen
shake and sound effects.

What is not right: the self-playing mode loses its lives within a few seconds.

**How often it works:** we sent this prompt six times with these settings. Three of the six games were playable; three had a mistake
that stopped the game (two script errors, and one where no enemies came). A long program written in one go can need one round of fixes:
the chat app's live preview shows the error, and **Ask to fix** sends it back to the model.

## The prompt

The chat app's own instructions (what the chat can show), then:

```text
Build a polished arcade space shooter called "Star Drift" as ONE self-contained HTML file (inline CSS and JavaScript only: no libraries, images, fonts or network requests).

- Full-window canvas with a three-layer parallax starfield and a dark nebula glow.
- The player's ship moves with the mouse or the arrow keys / WASD and fires with Space or the mouse button. It has 3 lives and a short shield after it is hit.
- Enemies come in waves with formations: drifters that weave, divers that swoop at the player, and turrets that fire aimed shots. Every 5th wave is a boss with a health bar and two attack patterns.
- Power-ups drop from destroyed enemies: spread shot, rapid fire, shield, and a smart bomb. Each has its own color and icon.
- Score, a combo multiplier for quick kills, and a best score kept in localStorage.
- Juice: particle explosions, screen shake, muzzle flashes, glowing neon colors, and simple sound effects made with the Web Audio API.
- A title screen, a pause key (P), and a game-over screen with "press Enter to play again".
- If the page URL ends with #autoplay, a simple AI flies the ship (dodges bullets, follows enemies, fires constantly) so the game plays itself.
```
