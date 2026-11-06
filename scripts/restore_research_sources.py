"""Restore official DDXPlus sources and exact recorded JSON byte identities.

Never changes a trusted digest, invents a source record, retrains a model,
or activates the optional candidate. Downloads are synthetic CC-BY research data.
"""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

ML = Path(__file__).resolve().parents[1]
BACKUP = ML / 'reports/source-restoration-backup'


def exact_json_bytes(content, expected):
    lf = content.replace(b'\r\n', b'\n')
    for candidate in [content, lf, lf.replace(b'\n', b'\r\n')]:
        if hashlib.sha256(candidate).hexdigest() == expected:
            assert json.loads(candidate) == json.loads(content)
            return candidate
    raise ValueError('Content cannot reproduce the recorded fingerprint by line-ending restoration; refusing to rewrite identity')


def fetch(path, url, md5, expected_size):
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        with urlopen(url, timeout=120) as response, part.open('wb') as output:
            shutil.copyfileobj(response, output)
        if part.stat().st_size != expected_size or hashlib.md5(part.read_bytes()).hexdigest() != md5:
            raise ValueError('Downloaded file failed official source checks')
        part.replace(path)
    if path.stat().st_size != expected_size or hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError('Existing source does not match official identity; not overwritten')


def main():
    raw = ML / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    downloads = ML / 'data/external/ddxplus-official'
    downloads.mkdir(parents=True, exist_ok=True)
    schema_path = ML / 'docs/candidate_feature_schema_v1.json'
    schema = json.loads(schema_path.read_bytes())
    policy = json.loads((ML / 'reports/candidate_calibration/recorded_results/uncertainty_policy/policy.json').read_bytes())
    restorations = {
        ML / 'models/v2_feature_schema.json': schema['source_sha256']['models/v2_feature_schema.json'],
        schema_path: policy['candidate_schema_sha256'],
    }
    record = {'purpose': __doc__, 'official_source': 'https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585',
              'licence': 'CC BY 4.0; attribution to the DDXPlus authors is required', 'restored_json': {}}
    for path, expected in restorations.items():
        original = path.read_bytes()
        restored = exact_json_bytes(original, expected)
        relative = path.relative_to(ML)
        if restored != original:
            backup = BACKUP / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(original)
            path.write_bytes(restored)
        record['restored_json'][str(relative)] = {'before': hashlib.sha256(original).hexdigest(), 'recorded_and_restored': expected,
                                                  'json_values_changed': False}
    metadata = downloads / 'release_evidences.json'
    fetch(metadata, 'https://ndownloader.figshare.com/files/40278013', '0660962abe0bbf99e1fb77188c253597', 120263)
    content = exact_json_bytes(metadata.read_bytes(), schema['source_sha256']['data/raw/release_evidences.json'])
    destination = raw / 'release_evidences.json'
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError('Existing raw metadata differs; not overwritten')
    destination.write_bytes(content)
    record['evidence_sha256'] = hashlib.sha256(content).hexdigest()
    (ML / 'reports/source-restoration.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    archive = downloads / 'release_train_patients.zip'
    print('Fetching complete official training archive (140.9 MB); no fabricated/truncated source file', flush=True)
    fetch(archive, 'https://ndownloader.figshare.com/files/40278019', 'f75a78420ccb061951daa4db2d62139b', 140923730)
    with zipfile.ZipFile(archive) as source:
        entries = [i for i in source.infolist() if not i.is_dir()]
        if len(entries) != 1 or Path(entries[0].filename).name not in {'release_train_patients', 'release_train_patients.csv'}:
            raise ValueError('Unexpected archive layout; refusing extraction')
        entry = entries[0]
        if entry.file_size > 10_000_000_000 or shutil.disk_usage(raw).free < entry.file_size * 2:
            raise ValueError('Insufficient safe space for the complete source')
        target = raw / 'release_train_patients'
        if not target.exists():
            part = raw / 'release_train_patients.part'
            with source.open(entry) as input_stream, part.open('wb') as output:
                shutil.copyfileobj(input_stream, output)
            if part.stat().st_size != entry.file_size:
                raise ValueError('Incomplete training-source extraction')
            part.replace(target)
        elif target.stat().st_size != entry.file_size:
            raise ValueError('Existing training file differs; not overwritten')
        with target.open('rb') as checked:
            training_hash = hashlib.file_digest(checked, 'sha256').hexdigest()
        record['training_source'] = {'archive_md5': 'f75a78420ccb061951daa4db2d62139b', 'archive_member': entry.filename,
            'uncompressed_bytes': entry.file_size, 'full_archive_extracted': True,
            'sha256': training_hash}
    (ML / 'reports/source-restoration.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('Official source restored; trusted digests and JSON meanings unchanged', flush=True)


if __name__ == '__main__':
    main()
