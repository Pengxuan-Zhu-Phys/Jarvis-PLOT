"""Read-only, stdlib-only source/wheel release gates; never publish or tag."""
import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import re
import zipfile


LIVE_DOCS = {
    'README.md': 'Current source candidate:',
}


def check_source(root):
    """Check the project table, VERSION, README version line and latest CHANGELOG entry."""
    text = (root/'pyproject.toml').read_text(encoding='utf-8')
    project = re.search(r'(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)', text)
    if project is None:
        raise ValueError('pyproject.toml: missing [project]')
    def field(name):
        match = re.search(rf'(?m)^{name}\s*=\s*[\"\x27]([^\"\x27]+)[\"\x27]\s*$', project[1])
        if match is None:
            raise ValueError(f'pyproject.toml: missing project.{name}')
        return match[1]
    if field('name') != 'JarvisPLOT':
        raise ValueError('Distribution name must be JarvisPLOT')
    version = field('version')
    if (root/'VERSION').read_text(encoding='utf-8').strip() != version:
        raise ValueError('VERSION and pyproject.toml disagree')
    changelog = (root/'CHANGELOG.md').read_text(encoding='utf-8')
    releases = re.findall(r'(?m)^##[ \t]+v?(\d+\.\d+\.\d+)(?=[ \t\u2014\u2013:-]|$)', changelog)
    if not releases or releases[0] != version:
        raise ValueError(f'CHANGELOG.md: latest release entry must be {version}')
    for relative, marker in LIVE_DOCS.items():
        lines = (root/relative).read_text(encoding='utf-8').splitlines()
        live = next((line for line in lines if line.startswith(marker)), '')
        if not re.search(rf'(?<![\d.]){re.escape(version)}(?![\d.])', live):
            raise ValueError(f'{relative}: live-version line must report {version}')
    return version


def check_wheel(root, wheel, version):
    """Compare every native module and declared resource with the built wheel."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_paths = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata_paths) != 1:
            raise ValueError('Wheel must contain one distribution metadata file')
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if metadata['Name'] != 'JarvisPLOT' or metadata['Version'] != version:
            raise ValueError('Wheel name/version disagrees with source')
        prefix = metadata_paths[0].rsplit('/', 1)[0]
        entry = archive.read(prefix+'/entry_points.txt').decode('utf-8')
        if 'jplot = jarvisplot.client:main' not in entry:
            raise ValueError('Wheel is missing the official jplot entry point')
        paths = list((root/'jarvisplot').rglob('*.py'))
        for relative in ('cards', 'Figure/cards', 'schema', 'manual_cards'):
            paths.extend(p for p in (root/'jarvisplot'/relative).rglob('*')
                         if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts))
        for path in set(paths):
            name = path.relative_to(root).as_posix()
            if name not in names:
                raise ValueError(f'Wheel is missing {name}')
            if archive.read(name) != path.read_bytes():
                raise ValueError(f'Wheel has stale content: {name}')
    return {'wheel': str(wheel), 'modules_and_resources': len(set(paths))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--wheel', type=Path, help='Built wheel to compare against the source')
    parser.add_argument('--tag', help='Optional release tag; must be VERSION or vVERSION')
    args = parser.parse_args(argv)
    try:
        version = check_source(args.root)
        if args.tag is not None and args.tag not in (version, 'v'+version):
            raise ValueError(f'Release tag {args.tag!r} does not match {version}')
        result = {'version': version, 'source': 'passed', 'publishes': False}
        if args.wheel:
            result.update(check_wheel(args.root, args.wheel, version))
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        parser.exit(1, f'Release check failed: {exc}\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
