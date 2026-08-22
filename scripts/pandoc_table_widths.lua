-- Give pipe tables explicit wrapping widths so they remain readable in PDF.
local dropped_title = false

function Header(header)
  if not dropped_title and header.level == 1 then
    dropped_title = true
    return {}
  end
  local title = pandoc.utils.stringify(header.content)
  if title:match("^31%. Oral%-defense") then
    return {pandoc.RawBlock("latex", "\\clearpage"), header}
  end
  return header
end

function Str(text)
  text.text = text.text:gsub("–", "-"):gsub("—", "-"):gsub("‑", "-")
  return text
end

function Code(code)
  code.text = code.text:gsub("–", "-"):gsub("—", "-"):gsub("‑", "-")
  return code
end

function CodeBlock(block)
  block.text = block.text:gsub("–", "-"):gsub("—", "-"):gsub("‑", "-")
  block.text = block.text:gsub("─", "-"):gsub("└", "+")
  return block
end

function Table(tbl)
  local layouts = {
    [2] = {0.28, 0.68},
    [3] = {0.22, 0.26, 0.48},
    [4] = {0.16, 0.20, 0.28, 0.32},
    [5] = {0.15, 0.18, 0.20, 0.21, 0.22},
    [6] = {0.15, 0.15, 0.16, 0.16, 0.16, 0.18},
  }
  local widths = layouts[#tbl.colspecs]
  if widths then
    for index, colspec in ipairs(tbl.colspecs) do
      tbl.colspecs[index] = {colspec[1], widths[index]}
    end
  end
  return tbl
end
