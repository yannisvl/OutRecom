import argparse
from classes.FacilityLocation import FacilityLocation

def main():
    parser = argparse.ArgumentParser(description='Neurips experiments')
    parser.add_argument('--problem', help='FL')
    parser.add_argument('--dataset', help='Brightkite, Gowalla, Twitter, Autotel, ')
    parser.add_argument('--confidence', type=float, help='Confidence arameter of Coordinatewise Median with Predictions')
    parser.add_argument('--unique', action='store_true', help='Keep only unique rows in the concatenated DataFrame')
    args = parser.parse_args()

    dataset = args.dataset
    keep_unique = args.unique
    c = args.confidence
    if args.problem == 'FL':
        prob = FacilityLocation(dataset, keep_unique, c)
    else:
        print("Invalid problem")
        exit()

    prob.run_experiment()
    
if __name__ == "__main__":
    main()