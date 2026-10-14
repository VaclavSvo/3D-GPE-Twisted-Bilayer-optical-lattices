import sys

from twisted_bilayer.cli import main

sys.exit(main(["gates", *sys.argv[1:]]))
