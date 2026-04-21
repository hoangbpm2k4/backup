#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/vietnam/voice_to_text"
REMOTE_BASE="gdrive:backupdatavoice"
LOG_FILE="$HOME/backup_drive.log"
MANIFEST_LOCAL="$ROOT/backup_manifest.txt"
TMP_ROOT="/tmp/backupdatavoice_shards"

RCLONE_FLAGS=(
  --transfers 4
  --checkers 8
  --tpslimit 10
  --drive-chunk-size 256M
  --retries 10
  --low-level-retries 20
)

# flatten_name|relative_source_dir
ITEMS=(
  "yt1900h_accepted|data/data_label_auto_yt_1900h/accepted"
  "yt_run_full_accepted|data/data_label_auto_yt_run_full/accepted"
  "yt_run_full_segments_denoised|data/data_label_auto_yt_run_full/segments_denoised"
  "overlap_yt_vlsp|data/overlap_zipformer_prep/full_overlap_yt_vlsp"
  "vlsp2023|data/VLSP2023/VLSP2023"
  "overlap_multidataset_yt1000h_overlap2|data/overlap_multidataset_yt1000h/overlap2"
  "overlap_multidataset_yt1000h_overlap3|data/overlap_multidataset_yt1000h/overlap3"
  "lhotse_cutsets_new_train|data/lhotse_cutsets_new_train"
)

mkdir -p "$TMP_ROOT"
mkdir -p "$ROOT"

exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

ensure_remote() {
  log "Ensuring remote dir exists: $REMOTE_BASE/"
  rclone mkdir "$REMOTE_BASE" "${RCLONE_FLAGS[@]}"
}

init_manifest() {
  if [[ ! -f "$MANIFEST_LOCAL" ]]; then
    {
      echo -e "timestamp\tname\tsource_dir\tcompressed_bytes\tsha256\tparts\tstatus"
    } > "$MANIFEST_LOCAL"
  fi
}

already_done() {
  local name="$1"
  grep -E $'\t'"$name"$'\t.*\tDONE$' "$MANIFEST_LOCAL" >/dev/null 2>&1
}

pack_to_shards() {
  local name="$1"
  local rel="$2"
  local workdir="$TMP_ROOT/$name"
  local sha_file="$workdir/${name}.sha256"
  local part_prefix="$workdir/${name}.tar.zst.part"

  mkdir -p "$workdir"

  if compgen -G "${part_prefix}[0-9][0-9][0-9][0-9]" >/dev/null && [[ -s "$sha_file" ]]; then
    log "Reusing existing shards for $name"
    return
  fi

  rm -f "${part_prefix}"* "$sha_file"
  log "Packing + compressing + splitting: $rel -> ${part_prefix}0000..."
  # Keep CPU pressure lower to avoid affecting other jobs.
  nice -n 10 bash -o pipefail -c "
    tar -cf - -C \"$ROOT\" \"$rel\" \
    | zstd -T0 -3 \
    | tee >(sha256sum | awk '{print \$1}' > \"$sha_file\") \
    | split -b 5G -d -a 4 - \"$part_prefix\"
  "
}

upload_shards() {
  local name="$1"
  local rel="$2"
  local workdir="$TMP_ROOT/$name"
  local sha_file="$workdir/${name}.sha256"
  local part_prefix="$workdir/${name}.tar.zst.part"

  if [[ ! -s "$sha_file" ]]; then
    log "ERROR: missing sha file for $name: $sha_file"
    return 1
  fi

  local -a parts=()
  mapfile -t parts < <(find "$workdir" -maxdepth 1 -type f -name "${name}.tar.zst.part[0-9][0-9][0-9][0-9]" | sort)
  if [[ "${#parts[@]}" -eq 0 ]]; then
    log "ERROR: no shard parts found for $name"
    return 1
  fi

  local sha
  sha="$(cat "$sha_file" | tr -d '\n\r')"
  local total_bytes
  total_bytes="$(du -cb "${parts[@]}" | awk 'END{print $1}')"

  log "Uploading ${#parts[@]} shards for $name (compressed bytes=$total_bytes)..."
  local idx=0
  for p in "${parts[@]}"; do
    idx=$((idx + 1))
    local base
    base="$(basename "$p")"
    local remote_file="$REMOTE_BASE/$base"
    log "Upload [$idx/${#parts[@]}]: $base"
    rclone copyto "$p" "$remote_file" "${RCLONE_FLAGS[@]}"
  done

  # Upload checksum sidecar (single-file compressed stream checksum)
  printf '%s  %s.tar.zst\n' "$sha" "$name" > "$workdir/${name}.tar.zst.sha256"
  rclone copyto "$workdir/${name}.tar.zst.sha256" "$REMOTE_BASE/${name}.tar.zst.sha256" "${RCLONE_FLAGS[@]}"

  # Record local manifest row and upload manifest after each dataset.
  printf '%s\t%s\t%s\t%s\t%s\t%s\tDONE\n' \
    "$(date '+%F %T')" \
    "$name" \
    "$rel" \
    "$total_bytes" \
    "$sha" \
    "${#parts[@]}" >> "$MANIFEST_LOCAL"
  rclone copyto "$MANIFEST_LOCAL" "$REMOTE_BASE/backup_manifest.txt" "${RCLONE_FLAGS[@]}"

  log "Uploaded $name successfully. Cleaning temp shards."
  rm -f "${parts[@]}" "$workdir/${name}.tar.zst.sha256" "$sha_file"
  rmdir "$workdir" 2>/dev/null || true
}

print_summary() {
  local total_uploaded
  total_uploaded="$(awk -F '\t' 'NR>1 && $7=="DONE"{s+=$4} END{print s+0}' "$MANIFEST_LOCAL")"
  local remote_count
  remote_count="$(rclone lsf "$REMOTE_BASE/" "${RCLONE_FLAGS[@]}" | wc -l | tr -d ' ')"
  log "Backup summary:"
  log "  total_uploaded_bytes=$total_uploaded"
  log "  remote_file_count=$remote_count"
  log "  remote_size:"
  rclone size "$REMOTE_BASE/" "${RCLONE_FLAGS[@]}"
}

main() {
  log "=== Backup start ==="
  log "ROOT=$ROOT"
  log "REMOTE=$REMOTE_BASE/"
  log "TMP_ROOT=$TMP_ROOT"
  init_manifest
  ensure_remote

  for item in "${ITEMS[@]}"; do
    IFS='|' read -r name rel <<<"$item"
    local_src="$ROOT/$rel"
    if [[ ! -d "$local_src" ]]; then
      log "SKIP missing source dir: $local_src"
      continue
    fi
    if already_done "$name"; then
      log "SKIP already marked DONE in manifest: $name"
      continue
    fi
    log "--- Processing: $name ($rel) ---"
    pack_to_shards "$name" "$rel"
    upload_shards "$name" "$rel"
  done

  # Ensure final manifest is present remotely.
  rclone copyto "$MANIFEST_LOCAL" "$REMOTE_BASE/backup_manifest.txt" "${RCLONE_FLAGS[@]}"
  print_summary
  log "=== Backup done ==="
}

main "$@"
