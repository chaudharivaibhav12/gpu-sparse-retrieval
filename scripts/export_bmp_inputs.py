"""Export a GSP subset and prepared cap32 queries to BMP's CIFF/PISA inputs."""
import argparse
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path
import struct
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def export(index, queries, directory, limit):
    if limit <= 0:
        raise ValueError('Query limit must be positive')
    if sys.byteorder != 'little':
        raise ValueError('This memoryview exporter requires a little-endian host')
    before = {str(p): digest(p) for p in (index, queries)}
    helper_path = Path(__file__).resolve().parents[1] / 'tests/check_cpu_pilot.py'
    spec = importlib.util.spec_from_file_location('ciff_fixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    directory.mkdir(parents=True, exist_ok=False)
    with index.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        if mapped[:8] != b'GSPIDX01':
            raise ValueError('Invalid GSP header')
        count, vocabulary_size, nonzeros = struct.unpack_from('<IIQ', mapped, 8)
        position = 24
        ids = memoryview(mapped)[position:position + count * 8].cast('Q')
        position += count * 8
        offsets = memoryview(mapped)[position:position + (count + 1) * 8].cast('Q')
        position += (count + 1) * 8
        terms = memoryview(mapped)[position:position + nonzeros * 4].cast('I')
        position += nonzeros * 4
        impacts = memoryview(mapped)[position:position + nonzeros]
        position += nonzeros
        vocabulary = []
        for _ in range(vocabulary_size):
            length, = struct.unpack_from('<I', mapped, position)
            position += 4
            word = mapped[position:position + length].decode('utf-8')
            position += length
            if not word or any(c.isspace() for c in word):
                raise ValueError('PISA requires nonempty vocabulary tokens without whitespace')
            vocabulary.append(word)
        if position != len(mapped) or offsets[0] != 0 or offsets[-1] != nonzeros:
            raise ValueError('Invalid GSP lengths')
        postings = [bytearray() for _ in vocabulary]
        frequencies = [0] * vocabulary_size
        previous = [0] * vocabulary_size
        for doc in range(count):
            if offsets[doc] > offsets[doc + 1]:
                raise ValueError('Unordered offsets')
            for posting in range(offsets[doc], offsets[doc + 1]):
                term = terms[posting]
                if term >= vocabulary_size:
                    raise ValueError('Invalid term ID')
                value = helper.integer(1, doc - previous[term]) + helper.integer(2, impacts[posting])
                postings[term].extend(helper.data(4, value))
                previous[term] = doc
                frequencies[term] += 1
        with (directory / 'subset.ciff').open('xb') as output:
            output.write(helper.frame(helper.integer(1, 1) + helper.integer(2, vocabulary_size)
                                      + helper.integer(3, count)))
            for term, word in enumerate(vocabulary):
                message = helper.data(1, word.encode()) + helper.integer(2, frequencies[term])
                output.write(helper.frame(message + postings[term]))
            for doc, original in enumerate(ids):
                output.write(helper.frame(helper.integer(1, doc)
                                          + helper.data(2, str(original).encode())))
        del ids, offsets, terms, impacts
    membership = []
    with queries.open() as source, (directory / 'queries.pisa').open('x') as output:
        for number, line in enumerate(source):
            if number >= limit:
                break
            fields = line.rstrip('\n').split('\t')
            name = fields[0]
            if not name or ':' in name or any(c.isspace() for c in name):
                raise ValueError('Invalid query ID')
            tokens = []
            seen = set()
            for item in fields[1:]:
                term, weight = map(int, item.split(':'))
                if term in seen or not 0 <= term < vocabulary_size or not 1 <= weight <= 32:
                    raise ValueError('Invalid prepared cap32 query')
                seen.add(term)
                tokens.extend([vocabulary[term]] * weight)
            membership.append(name)
            output.write(name + ': ' + ' '.join(tokens) + '\n')
    if len(membership) != limit:
        raise ValueError('Insufficient prepared queries')
    if before != {str(p): digest(p) for p in (index, queries)}:
        raise ValueError('Input changed during export')
    manifest = {'documents': count, 'terms': vocabulary_size, 'nonzeros': nonzeros,
                'query_ids_in_order': membership, 'inputs': before,
                'query_policy': 'Already transformed cap32 weights expanded into repeated known vocabulary tokens; no second rescaling.',
                'outputs': {p.name: digest(p) for p in directory.iterdir() if p.is_file()}}
    (directory / 'export-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=25)
    args = parser.parse_args()
    export(args.index, args.queries, args.output, args.limit)


if __name__ == '__main__':
    main()
