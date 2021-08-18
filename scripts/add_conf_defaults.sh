#!/bin/bash

cd "$(dirname "$0")"

# first show config
mspcrunner config show

# =============================================================
# quant
# =============================================================
echo "adding quant"
mspcrunner config set-quant LF ../params/MASIC-LF-10ppm.xml
mspcrunner config set-quant TMT10 ../params/MASIC-TMT10-10ppm-ReporterTol0.003Da.xml
mspcrunner config set-quant TMT11 ../params/MASIC-TMT11-10ppm-ReporterTol0.003Da.xml
mspcrunner config set-quant TMT16 ../params/MASIC-TMT16-10ppm-ReporterTol0.003Da.xml

mspcrunner config show

# =============================================================
# search
# =============================================================
echo "adding search"

mspcrunner config set-search LF-OTIT ../params/MSFragger-LF-OTIT.conf

mspcrunner config set-search LF-OTOT ../params/MSFragger-LF-OTOT.conf
mspcrunner config set-search TMT10-11-OTOT ../params/MSFragger-TMT10-OTOT.params
mspcrunner config set-search TMT10-11-OTOT-phos ../params/MSFragger-TMT10-OTOT-phos.params
mspcrunner config set-search TMT16-OTOT ../params/MSFragger-TMT16-OTOT.params
mspcrunner config set-search TMT16-OTOT-phos ../params/MSFragger-TMT16-OTOT-phos.params


mspcrunner config show

