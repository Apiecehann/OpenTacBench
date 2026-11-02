"""SHA512-checked vcpkg downloads, with an optional GitHub transport prefix."""
import argparse
import hashlib
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--github-prefix',default='')
    parser.add_argument('url')
    parser.add_argument('sha512')
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    urls=[args.url]
    if args.github_prefix and args.url.startswith('https://github.com/'):
        urls.insert(0,args.github_prefix+args.url)
    for url in urls:
        result=subprocess.run(['curl','-L','--fail','--connect-timeout','12',
            '--max-time','90','--retry','1','-sS',url,'-o',str(args.destination)])
        if result.returncode==0:
            digest=hashlib.sha512(args.destination.read_bytes()).hexdigest()
            if digest==args.sha512:
                return
            print('SHA512 mismatch:',args.url)
    raise SystemExit('Could not obtain the expected archive: '+args.url)


if __name__=='__main__':
    main()
