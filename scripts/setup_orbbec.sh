#!/usr/bin/env bash
# Download and unpack Orbbec K4A Wrapper into third_party/orbbec/ (linux-64).
# pyk4a is installed by pixi; this script provides the native SDK libraries.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/orbbec"
VERSION="${ORBBEC_K4A_VERSION:-v1.10.4}"
ARCH="${ORBBEC_K4A_ARCH:-linux}"

mkdir -p "${DEST}"
cd "${DEST}"

if [[ -L current && -d current ]]; then
  echo "Orbbec K4A Wrapper already linked at ${DEST}/current"
  exit 0
fi

# Release asset names vary; try the Femto Bolt v1.10.4 linux bundle first.
CANDIDATES=(
  "OrbbecSDK_K4A_Wrapper_v1.10.4_linux_202504141102.tar.gz"
  "OrbbecSDK_K4A_Wrapper_v1.10.4_linux_202504141102.zip"
  "OrbbecSDK_K4A_Wrapper_${VERSION#v}_linux.tar.gz"
  "OrbbecSDK_K4A_Wrapper_${VERSION#v}_${ARCH}.zip"
)

BASE="https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases/download/${VERSION}"
ARCHIVE=""
for name in "${CANDIDATES[@]}"; do
  url="${BASE}/${name}"
  if curl -fsSL --head "${url}" >/dev/null 2>&1; then
    ARCHIVE="${name}"
    break
  fi
done

if [[ -z "${ARCHIVE}" ]]; then
  echo "Could not find a ${VERSION} ${ARCH} release asset on GitHub." >&2
  echo "Set ORBBEC_ROOT manually or download from:" >&2
  echo "  https://github.com/orbbec/OrbbecSDK-K4A-Wrapper/releases" >&2
  exit 1
fi

echo "Downloading ${BASE}/${ARCHIVE} ..."
curl -fsSL -o "${ARCHIVE}" "${BASE}/${ARCHIVE}"

case "${ARCHIVE}" in
  *.tar.gz) tar -xzf "${ARCHIVE}" ;;
  *.zip) unzip -q "${ARCHIVE}" ;;
  *) echo "Unknown archive type: ${ARCHIVE}" >&2; exit 1 ;;
esac

EXTRACTED="$(find . -maxdepth 1 -type d -name 'OrbbecSDK_K4A_Wrapper_*' | head -1)"
if [[ -z "${EXTRACTED}" ]]; then
  echo "Extracted archive but no OrbbecSDK_K4A_Wrapper_* directory found." >&2
  exit 1
fi

ln -sfn "$(basename "${EXTRACTED}")" current
rm -f "${ARCHIVE}"
echo "Orbbec K4A Wrapper ready: ${DEST}/current"

bash "${ROOT}/scripts/install_pyk4a.sh"
echo "Re-enter pixi shell so PATH/LD_LIBRARY_PATH update."
