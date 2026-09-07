# Hybrid mark

Two complementary, rotationally symmetric forms share a stepped diagonal
negative space. The mark echoes opposing agents and the combination of learned
policy and physical search, without depicting a tank, maze, or letter.

`hybrid.svg` is the vector master: a 32-unit grid, two filled paths, no fonts,
embedded bitmaps, strokes, filters, or gradients. Its 24-unit silhouette keeps
the diagonal opening visible at 16px. Charcoal `#242523` switches to bone
`#f3f2ed` with the browser's dark color preference.

PNG derivatives are rendered directly from the paths. The 16/32px fallbacks
use a bone rounded tile so they remain visible in either browser theme.
The 180/192/512px install icons use a full-bleed bone background with the mark
scaled to 75% around the center to stay within the maskable safe zone.

The `hybrid-*` filenames deliberately differ from the retired screenshot icons
to avoid reusing previously cached favicons. Only Hybrid assets are linked by
the entry pages and web app manifest.
