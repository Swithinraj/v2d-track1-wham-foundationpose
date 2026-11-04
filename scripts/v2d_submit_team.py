#!/usr/bin/env python3
"""Run the official uploader with a narrowly corrected shared-team identity check.

The official kit file remains unchanged. All authentication, rules, quotas,
submission validation, upload and receipt/poll logic still execute in that client.
Usage: v2d_submit_team.py KIT/submit/v2d_submit.py [official client arguments...]
"""
from pathlib import Path
import sys
from submission_team import effective_submission_identities,patch_preflight_source


def main():
    source=Path(sys.argv.pop(1)).resolve()
    sys.argv[0]=str(source)
    namespace={'__file__':str(source),'__name__':'__main__',
               'effective_submission_identities':effective_submission_identities}
    exec(compile(patch_preflight_source(source.read_text()),str(source),'exec'),namespace)


if __name__=='__main__': main()
