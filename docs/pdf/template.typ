// Typst template for docs/LEARNING_GUIDE.pdf.
// Used by scripts/build_pdf.py: pandoc inserts the converted guide where the body variable is.

#let accent = rgb("#1c5cab")
#let ink = rgb("#0b0b0b")
#let muted = rgb("#52514e")
#let rule-color = rgb("#d9d8d2")
#let sans = ("Segoe UI", "Helvetica Neue", "Arial", "Libertinus Serif")
#let serif = ("Libertinus Serif", "Nirmala UI")
#let mono = ("DejaVu Sans Mono", "Nirmala UI")
#let guide-title = "$title$"

#set document(title: guide-title)
#set text(font: serif, size: 10.5pt, fill: ink, lang: "en")
#set par(justify: true, leading: 0.68em, spacing: 1.05em)
#set page(paper: "a4", margin: (x: 2.3cm, top: 2.5cm, bottom: 2.4cm))

// ---- pandoc helpers -------------------------------------------------------
#let horizontalrule = line(length: 100%, stroke: 0.5pt + rule-color)
#show terms.item: it => block(breakable: false)[
  #text(weight: "bold")[#it.term]
  #block(inset: (left: 1.5em, top: -0.4em))[#it.description]
]

// ---- code -----------------------------------------------------------------
#show raw: set text(font: mono, size: 8pt)
#show raw.where(block: true): it => block(
  width: 100%,
  fill: rgb("#f7f7f4"),
  stroke: (left: 2pt + rgb("#c9c8c2")),
  inset: (x: 9pt, y: 8pt),
  radius: 2pt,
  breakable: true,
  {
    set par(justify: false, leading: 0.55em)
    it
  },
)
#show raw.where(block: false): box.with(
  fill: rgb("#f1f1ee"),
  inset: (x: 2.5pt, y: 0pt),
  outset: (y: 2.5pt),
  radius: 2pt,
)

// Caption above a snippet copied from the repository: file name and line numbers.
#let codecaption(file, where) = block(sticky: true, above: 1.2em, below: 0.45em)[
  #set text(font: sans, size: 8.5pt)
  #box(fill: accent, inset: (x: 5pt, y: 2.5pt), radius: 2pt)[
    #text(fill: white, weight: "bold", font: mono, size: 7.8pt)[#file]
  ]
  #h(5pt)
  #text(fill: muted)[#where]
]

// ---- boxes ("Key ideas", "Note", "Detail you can ignore for now") ----------
#let callout(kind: "note", title: "", body) = {
  let styles = (
    key: (rgb("#eaf2fd"), rgb("#2a78d6")),
    note: (rgb("#f4f4f1"), rgb("#8a897f")),
    skip: (rgb("#f1effc"), rgb("#6250d6")),
    warn: (rgb("#fdf0ea"), rgb("#d95926")),
  )
  let (background, bar) = styles.at(kind)
  block(
    width: 100%,
    fill: background,
    stroke: (left: 3pt + bar),
    inset: (x: 11pt, y: 10pt),
    radius: 2pt,
    breakable: true,
    above: 1.2em,
    below: 1.2em,
  )[
    #text(font: sans, weight: "bold", size: 8.5pt, fill: bar.darken(25%), tracking: 0.5pt)[#upper(title)]
    #v(-0.3em)
    #set par(justify: false)
    #body
  ]
}

// Ordinary block quotes (not boxes).
#show quote.where(block: true): it => block(
  inset: (left: 1.1em, y: 0.2em),
  stroke: (left: 2pt + rule-color),
  text(style: "italic", it.body),
)

// ---- tables and figures ----------------------------------------------------
#set table(inset: (x: 7pt, y: 5pt), stroke: none)
#show table: set text(size: 9.5pt)
#show table: set par(justify: false)
#show table.cell.where(y: 0): set text(weight: "bold", font: sans, size: 8.8pt)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: image): set figure.caption(position: bottom)
#show figure.caption: it => block(width: 92%)[
  #set text(size: 9pt, fill: muted)
  #set par(justify: false)
  #it
]
#set figure(gap: 0.9em)
#show figure: set block(above: 1.4em, below: 1.4em)
#set image(width: 92%)

// ---- links, lists, math ------------------------------------------------------
#show link: set text(fill: accent)
#set list(indent: 0.6em, body-indent: 0.6em)
#set enum(indent: 0.6em, body-indent: 0.6em)
#show math.equation.where(block: true): set block(above: 1.1em, below: 1.1em)

// ---- headings --------------------------------------------------------------
#show heading: set text(font: sans, fill: ink)
#show heading: set par(justify: false)
#show heading: set block(sticky: true)

// Level 1 = a Part, an Appendix or a front section. (The page break before it is
// added by filter.lua, so that a Part can have a divider page of its own.)
#show heading.where(level: 1): it => {
  v(1.2cm)
  block(below: 1.1cm)[
    #text(size: 25pt, weight: "bold", fill: accent)[#it.body]
    #v(-0.3cm)
    #line(length: 4cm, stroke: 2.5pt + accent)
  ]
}
// A Part gets a whole page to itself, with the title a third of the way down.
#let part-heading(body) = page(header: none)[
  #v(1fr)
  #heading(level: 1, body)
  #v(2fr)
]
// Level 2 = a Chapter: also starts a new page.
#show heading.where(level: 2): it => {
  pagebreak(weak: true)
  block(above: 0.4cm, below: 0.8cm)[
    #text(size: 19pt, weight: "bold")[#it.body]
  ]
}
#show heading.where(level: 3): it => block(above: 1.6em, below: 0.8em)[
  #text(size: 12.5pt, weight: "bold")[#it.body]
]
#show heading.where(level: 4): it => block(above: 1.3em, below: 0.7em)[
  #text(size: 10.5pt, weight: "bold", fill: muted)[#it.body]
]

// ---- title page ------------------------------------------------------------
#page(margin: (x: 2.6cm, y: 3cm))[
  #set par(justify: false)
  #v(3.2cm)
  #line(length: 5cm, stroke: 4pt + accent)
  #v(0.5cm)
  #text(font: sans, size: 38pt, weight: "bold")[The Mini-GPT \ Learning Guide]
  #v(0.5cm)
  #text(font: sans, size: 15pt, fill: muted)[
    Build a GPT-style language model from scratch in PyTorch: \
    from an empty folder to a model that writes Shakespeare.
  ]
  #v(1.4cm)
  #block(width: 78%)[
    #set text(size: 11.5pt)
    Tokenizers, embeddings, self-attention, the transformer block, training and
    generation, each explained from first principles for a reader who knows Python
    but has never used NumPy, PyTorch or a neural network.
  ]
  #v(1fr)
  #text(font: sans, size: 10.5pt, fill: muted)[
    Companion to the #text(font: mono, size: 9.5pt)[mini-gpt-from-scratch] repository \
    $date$
  ]
]

// ---- table of contents ------------------------------------------------------
#show outline.entry.where(level: 1): it => {
  v(0.9em, weak: true)
  text(font: sans, weight: "bold", fill: accent, it)
}
#page[
  #text(font: sans, size: 25pt, weight: "bold", fill: accent)[Contents]
  #v(0.6cm)
  #outline(title: none, depth: 2, indent: 1.2em)
]

// ---- running header and page numbers (start counting at the first real page) --
#counter(page).update(1)
#set page(
  header: context {
    let here-page = here().page()
    // chapters (level 2) and parts / appendices (level 1): the header names whichever came last
    let chapters = query(heading.where(level: 1).or(heading.where(level: 2)))
    let on-page = chapters.filter(h => h.location().page() == here-page)
    let earlier = chapters.filter(h => h.location().page() < here-page)
    let current = if on-page.len() > 0 { on-page.first() } else if earlier.len() > 0 { earlier.last() } else { none }
    set text(font: sans, size: 8pt, fill: muted)
    grid(
      columns: (1fr, auto),
      if current != none { current.body },
      guide-title,
    )
    v(-0.5em)
    line(length: 100%, stroke: 0.4pt + rule-color)
  },
  footer: context align(center, text(font: sans, size: 9pt, fill: muted, counter(page).display("1"))),
)

$body$
