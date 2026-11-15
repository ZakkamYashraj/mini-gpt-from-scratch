-- Pandoc Lua filter used when building the learning-guide PDF (see scripts/build_pdf.py).
-- It turns three plain-markdown conventions into nicely styled Typst elements,
-- so the same LEARNING_GUIDE.md reads well on GitHub AND in the PDF.

local function typst(code)
  return pandoc.RawBlock("typst", code)
end

local function escape(text)
  return (text:gsub("\\", "\\\\"):gsub('"', '\\"'))
end

-- 1) A block quote that starts with a bold title becomes a coloured box:
--        > **Key ideas**
--        >
--        > - first point
function BlockQuote(el)
  local first = el.content[1]
  if not first or (first.t ~= "Para" and first.t ~= "Plain") then return nil end
  local head = first.content[1]
  if not head or head.t ~= "Strong" then return nil end

  local title = pandoc.utils.stringify(head)
  local kind = "note"
  if title:match("^Key ideas") then
    kind = "key"
  elseif title:match("ignore") then
    kind = "skip"
  elseif title:match("mistake") or title:match("Warning") or title:match("Careful") then
    kind = "warn"
  end

  -- drop the title (and the whitespace after it) from the first paragraph
  first.content:remove(1)
  while first.content[1] and (first.content[1].t == "Space" or first.content[1].t == "SoftBreak"
      or first.content[1].t == "LineBreak") do
    first.content:remove(1)
  end
  if #first.content == 0 then el.content:remove(1) end

  local blocks = { typst('#callout(kind: "' .. kind .. '", title: "' .. escape(title) .. '")[') }
  for _, block in ipairs(el.content) do blocks[#blocks + 1] = block end
  blocks[#blocks + 1] = typst("]")
  return blocks
end

-- 2) The caption that scripts/build_guide.py puts above each repository snippet,
--        **`mini_gpt/model.py` · lines 88–120**
--    becomes a small label that stays attached to the code block below it.
-- 3) A paragraph containing only an image becomes a numbered figure with a caption.
function Para(el)
  if #el.content == 1 and el.content[1].t == "Strong" then
    local inner = el.content[1].content
    if inner[1] and inner[1].t == "Code" then
      local file = inner[1].text
      local rest = pandoc.List()
      for i = 2, #inner do rest:insert(inner[i]) end
      local where = pandoc.utils.stringify(rest):gsub("^%s*·%s*", "")
      return typst('#codecaption("' .. escape(file) .. '", "' .. escape(where) .. '")')
    end
  end
  if #el.content == 1 and el.content[1].t == "Image" then
    local image = el.content[1]
    local caption = pandoc.Plain(image.caption)
    image.caption = {}
    return pandoc.Figure({ pandoc.Plain({ image }) }, { caption })
  end
  return nil
end

-- 4) Level-1 headings start a new page; a "Part N: ..." heading gets a divider page.
function Header(el)
  if el.level ~= 1 then return nil end
  local title = pandoc.utils.stringify(el.content)
  if title:match("^Part %d") then
    return typst('#part-heading[' .. escape(title) .. ']')
  end
  return { typst("#pagebreak(weak: true)"), el }
end
