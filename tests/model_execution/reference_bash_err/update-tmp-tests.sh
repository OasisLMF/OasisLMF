readarray -t files < <(find . -maxdepth 1 -type f -name "*partition*.sh")

for f in "${files[@]}"; do
    echo $f
    sed -i \
        -e 's|/tmp/[^ ]*/fifo/|/tmp/%FIFO_DIR%/fifo/|g' \
        -e '/rm -R -f/ s|/tmp/[^ ]*/|/tmp/%FIFO_DIR%/|g' \
        $f
done
