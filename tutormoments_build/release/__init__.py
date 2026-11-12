"""Build the public data release from the de-identified bucket.

Everything lives under one release prefix in the bucket, never on a laptop::

    <release>/
      snapshots/<snapshot_id>/   frozen copies of the inputs + snapshot.json (hashes)
      builds/<build_id>/         one folder per build, laid out like the HF repo
                                 + manifest.json (file hashes, commit, snapshot)

A snapshot is written once and never overwritten, so a build is reproducible
from (snapshot_id, code commit). The nightly annotation export keeps moving;
builds never read it directly.

The bucket comes from ``--root`` or ``TUTORMOMENTS_RELEASE_ROOT``
(``s3://<bucket>`` or a local directory for tests); it is never written in code.
Only the de-identified bucket is a valid root.
"""
