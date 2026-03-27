import re, sys

with open("oncogene_promoters_raw.fa") as fin, \
     open("oncogene_promoters.fa", "w") as fout:
    for line in fin:
        if line.startswith(">"):
            # extract chr and coordinates from UCSC header
            m = re.search(r'range=(chr\w+):([0-9,]+)-([0-9,]+)', line)
            if m:
                chrom = m.group(1)
                start = m.group(2).replace(',', '')
                end   = m.group(3).replace(',', '')
                gene  = line.split()[0].lstrip('>')
                fout.write(f">hg38|{chrom}|{start}-{end}\n")
            else:
                fout.write(line)
        else:
            fout.write(line)