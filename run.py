import argparse
import json
from pathlib import Path
from src.geometry import make_bracket, evaluate_points


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd', required=True)
    g=sub.add_parser('generate'); g.add_argument('--noise-mm',type=float,default=0.0)
    e=sub.add_parser('evaluate'); e.add_argument('reference'); e.add_argument('predicted')
    i=sub.add_parser('interpret'); i.add_argument('image'); i.add_argument('--config',default='config/settings.yaml')
    args=p.parse_args()
    if args.cmd=='generate':
        print('Generated:', make_bracket(noise_sigma_mm=args.noise_mm))
    elif args.cmd=='evaluate':
        print(json.dumps(evaluate_points(args.reference,args.predicted),indent=2))
    else:
        from src.model import interpret_image
        print(json.dumps(interpret_image(args.image,args.config),indent=2))

if __name__=='__main__': main()
