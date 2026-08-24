Place the two institutional logos here:

  lassonde.png   Lassonde School of Engineering logo
  yorku.png      York University logo

PNG with a transparent background is preferred; PDF or JPG also work, but then
update the file extensions in the \IfFileExists / \includegraphics calls on the
title page in main.tex.

They are rendered side by side at 1.5 cm height at the top of the title page.
Until the files are present, main.tex silently omits them so the build still
succeeds.
