# RakshyaNet — 2-minute demo video script

> **This is a screen recording of the running product. No slides, no title
> cards, no deck.** Every second of footage is the live app being driven with a
> mouse. The only thing on screen is RakshyaNet itself.

**Target: 1:55–2:00.** ~285 spoken words at ~150 wpm.
Voiceover over screen capture — you narrate while you click.

---

## Before you hit record

1. Backend running on `:8000`, frontend on `:5173`.
2. Browser at **1600×1000**, zoom 100%, **F11 fullscreen** so no tabs or
   bookmarks show. Dismiss any browser notification popups first.
3. Press **Start fresh** in the header and let it finish — the demo must open
   clean, with no leftover road closure from a previous take.
4. Open the **Operations** tab. That is frame one.
5. Do one silent dry run of the click path. The clicks must land on the beat;
   nothing kills a demo video like hunting for a button on camera.

**Click path, start to finish:**
Operations → Gemma evidence → (hover `UNKNOWN`, hover a citation chip) → Agent
console → (let the reasoning render) → Math lab → (urgency plinth, baseline
table) → Review & authorize → hold.

---

### [0:00–0:18] — Hook

**Do:** Start on **Operations**. Slowly orbit/pan the 3D Nepal terrain once.
Incidents are already plotted. Don't click anything yet.

> In 2015, I was a child in Karnali. Our home was cut off.
>
> And what I remember is not that there was nothing — it's that the supplies
> existed *somewhere else*, and never reached us.
>
> Nepal doesn't have a scarcity problem in a disaster. It has a coordination
> problem.

---

### [0:18–0:40] — The real problem

**Do:** Click **Gemma evidence**. Scroll so the police report and the
municipality report are visible together.

> Here's what a dispatcher actually gets. Three reports, from three people who
> never spoke to each other.
>
> One says heavy vehicles cannot pass. Another says motorcycles still can.
>
> Both are credible. Average them into "blocked" and you delete a road
> motorcycles could use — you strand a district. Average them into "open" and
> you drive a truck into a landslide.
>
> The hard part was never the routing. It's the *interpretation*.

---

### [0:40–1:08] — Gemma reads it, and admits what it doesn't know

**Do:** Stay on **Gemma evidence**. Hover the extracted fields so the confidence
values show. **Hover `medical urgency` — the `UNKNOWN` one.** Then hover a
citation chip so the evidence ID it points to highlights.

> So Gemma reads them, and returns one structured object — where every single
> field cites the exact report it came from.
>
> And look at this one. Medical urgency: **unknown**. No source mentioned
> injuries, so Gemma doesn't guess.
>
> And we don't take the citations on trust either — we check them in code. If it
> cites a source that doesn't exist, we reject the whole analysis.

---

### [1:08–1:32] — Native function calling *(the most important 20 seconds)*

**Do:** Open the **Agent console**. Let the recorded turn replay — the tool calls
appear first, then Gemma's own reasoning types out. **Stop talking for ~2 seconds
and let the reasoning be readable on screen.**

> Then Gemma runs the engine itself, through native function calling.
>
> It called `list_corridor_status` first — it wanted the real road graph before
> naming anything. Then it ran the optimization with **zero** corridors closed.
>
> And this is its own reasoning, verbatim: the road is *restricted*, not
> *established as impassable*. It refused to delete a usable corridor on
> contradictory evidence.
>
> It can't invent one either — every corridor ID it emits is checked against the
> graph before the engine ever sees it.

---

### [1:32–1:50] — The bound, and the proof

**Do:** Click **Math lab**. Point at the urgency plinth
(`need + survival + Gemma = total`), then scroll to the **baseline comparison
table**.

> Gemma's influence on priority is capped at **one point zero** — against a
> survival penalty of **ten**, computed from measured stock and population.
>
> It can move an incident up the queue. It cannot outweigh a real shortage.
>
> And against a naive shortest-path baseline, when the main east–west corridor
> closes: naive strands **all five** trucks. We keep all five driving, for eight
> percent more distance.

---

### [1:50–2:00] — The gate

**Do:** Click **Review & authorize**. The blocker naming the unknown field
should be visible. Hold the final frame for 2 seconds after you stop speaking.

> And then it stops. Gemma cannot allocate, route, approve, or dispatch.
>
> A human signs — and if a required field is still unknown, it says so *by name*
> before anyone can.

---

## Delivery notes

- **Never cut 1:08–1:32.** Gemma refusing to close the corridor is the whole
  submission. If you run long, trim the opening pan instead.
- Say **"unknown"** and **"it refused"** clearly. Those two moments are what
  separate this from a chatbot with a map on it.
- Do **not** say "the AI decides". It doesn't, and the entire architecture exists
  to make that checkable.
- If a live Gemma call is slow on the day, don't wait on camera — the agent
  console replays the **recorded** turn, which is what this script uses anyway,
  and it is labelled as a replay on screen, so it stays honest.
- **Optional 3-second add** if you're under time: on Operations, drop a road
  closure and let the plan visibly re-route. It is the strongest single visual in
  the product.

---

## If you want to show the satellite tool (only if you have spare seconds)

It is **off by default**, so it will not appear unless you enable the flag. If
you do show it, say this and nothing more — it is easy to overclaim:

> Gemma can also ask for an overhead satellite read of a corridor — and one of
> its triggers is a weather advisory, so it can check a road *before* anyone has
> reported it blocked. But imagery can only ever corroborate a closure. It can
> never cause one.
