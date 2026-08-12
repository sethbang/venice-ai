// pydoc-markdown de-indents RST literal blocks ("Example::" + indented code/ASCII) and
// leaves `>>>` doctests at inconsistent indentation, so Docusaurus/CommonMark flows them
// into prose. This re-wraps both in fenced code blocks. Runs after prune, on the
// generated (gitignored) API markdown only.
import {readdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';

const API = 'docs/api';

// A structural element pydoc emits at column 0 that ends a block.
// >=2 hashes for headers so a Python "# comment" line doesn't end a block.
const BOUNDARY = /^(#{2,6}\s|\*\*[A-Za-z][A-Za-z ]*\*\*|<a[ >]|```|-{3,}\s*$|={3,}\s*$)/;
const INTRO = /^(.*\S)::\s*$/;          // "Foo::" RST literal-block intro
const DOCTEST = /^\s*>>>(\s|$)/;        // ">>> ..." doctest prompt
const CONT = /^\s*\.\.\.(\s|$)/;        // "... " doctest continuation

function stripCommonIndent(block) {
  const widths = block.filter((l) => l.trim() !== '').map((l) => l.match(/^(\s*)/)[1].length);
  const min = widths.length ? Math.min(...widths) : 0;
  return min ? block.map((l) => l.slice(min)) : block;
}

function fixContent(text) {
  const lines = text.split('\n');
  const out = [];
  let i = 0;

  // Preserve YAML front-matter verbatim.
  if (lines[0] === '---') {
    out.push(lines[i++]);
    while (i < lines.length && lines[i] !== '---') out.push(lines[i++]);
    if (i < lines.length) out.push(lines[i++]);
  }

  let inFence = false;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {inFence = !inFence; out.push(line); i++; continue;}
    if (inFence) {out.push(line); i++; continue;}

    // RST literal block: "Foo::"
    const m = line.match(INTRO);
    if (m) {
      out.push(m[1] + ':');
      i++;
      if (i < lines.length && lines[i].trim() === '') i++;
      const block = [];
      while (i < lines.length) {
        const l = lines[i];
        if (BOUNDARY.test(l) || INTRO.test(l)) break;
        if (l.trim() === '' && block.length && block[block.length - 1].trim() === '') break;
        block.push(l);
        i++;
      }
      while (block.length && block[block.length - 1].trim() === '') block.pop();
      if (block.length) out.push('', '```', ...stripCommonIndent(block), '```', '');
      continue;
    }

    // Doctest block: a ">>>" line (possibly indented).
    if (DOCTEST.test(line)) {
      const block = [];
      while (i < lines.length) {
        const l = lines[i];
        if (BOUNDARY.test(l) || INTRO.test(l)) break;
        if (l.trim() === '') {
          // Look past blank line(s): keep the block going if the next real line is
          // another prompt/continuation or an interspersed "# comment"; otherwise stop.
          let j = i + 1;
          while (j < lines.length && lines[j].trim() === '') j++;
          const next = lines[j] ?? '';
          if (j >= lines.length || (!DOCTEST.test(next) && !CONT.test(next) && !/^\s*#/.test(next))) break;
        }
        block.push(l);
        i++;
      }
      while (block.length && block[block.length - 1].trim() === '') block.pop();
      if (block.length) out.push('', '```python', ...stripCommonIndent(block), '```', '');
      continue;
    }

    out.push(line);
    i++;
  }
  return out.join('\n');
}

function walk(dir) {
  for (const e of readdirSync(dir, {withFileTypes: true})) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.md')) {
      const before = readFileSync(p, 'utf8');
      const after = fixContent(before);
      if (after !== before) writeFileSync(p, after);
    }
  }
}

walk(API);
console.log('Re-fenced RST literal blocks and doctests in API docs.');
