#from inclearn import parser
#from inclearn.train import train

import parser
from train import train

args = parser.get_parser().parse_args()
args = vars(args)  # Converting argparse Namespace to a dict.

if args["seed_range"] is not None:
    args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
    print("Seed range", args["seed"])

train(args)
