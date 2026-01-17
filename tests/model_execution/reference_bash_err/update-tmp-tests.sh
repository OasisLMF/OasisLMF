readarray -t files < <(find . -maxdepth 1 -type f -name "*partition*.sh")

for f in "${files[@]}"; do
    echo $f
    sed -i 's|/tmp/[a-zA-Z0-9]*/|/tmp/%FIFO_DIR%/|g' $f
done
