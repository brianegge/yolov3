#!/bin/bash

for f in "$@"
do
  base=$(basename "$f")
  output=$(dirname "$f")/$(basename "$f" .jpg).xml
  size=$(file "$f" | grep -Eo "[0-9]{2,}x[0-9]+")
  width=$(echo $size | cut -dx -f1)
  height=$(echo $size | cut -dx -f2)
  echo $f=$output
  cat > "$output" <<EOF
<annotation>
	<folder></folder>
	<filename>${base}</filename>
	<path>${base}</path>
	<source>
		<database>roboflow.ai</database>
	</source>
	<size>
		<width>$width</width>
		<height>$height</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
</annotation>
EOF
done
