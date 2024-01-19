import json
import numpy
import pandas as pd
import argparse
import sys
import time
import collections

BUFFER_SIZE = 10 # use even number for simple dividing in half
THRESHOLD_LOW = 3
THRESHOLD_HIGH = 7

def get_zero_crossings(df):

    start_time = time.time()
    flight_dict = {}
    zero_crossings = []

    for index, row in df.iterrows():
        hex_code = row["hex"]

        # Skip aircraft on the ground
        if row["alt_baro"] == "ground":
            continue

        # If first time seen, add buffer
        if hex_code not in flight_dict:
            flight_dict[hex_code] = collections.deque(maxlen=10)
            flight_dict[hex_code].append(row["nic"])
        
        flight_dict[hex_code].append(row["nic"])

        # If buffer is full start checking for zero crossings
        if hex_code in flight_dict and len(flight_dict[hex_code]) == BUFFER_SIZE:
            split_index = int(BUFFER_SIZE/2)
            prior_window = list(flight_dict[hex_code])[:split_index]
            later_window = list(flight_dict[hex_code])[split_index:]
            prior_window_avg = sum(prior_window) / len(prior_window)
            later_window_avg = sum(later_window) / len(later_window)
            gain_of_gps = prior_window_avg < THRESHOLD_LOW and later_window_avg >= THRESHOLD_HIGH
            loss_of_gps = prior_window_avg >= THRESHOLD_HIGH and later_window_avg < THRESHOLD_LOW
            if gain_of_gps or gain_of_gps:
                zero_crossing = {
                    "hex_code": hex_code,
                    "alt_geom": row["alt_geom"],
                    "row_index": index,
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "prior_nic_window": prior_window_avg,
                    "later_nic_window": later_window_avg,
                    "window": list(flight_dict[hex_code])
                }
                zero_crossings.append(zero_crossing)
            
            ## Flush buffer
            flight_dict[hex_code] = collections.deque(maxlen=10)

    end_time = time.time()
    print("Processed file:", (end_time - start_time), "seconds")
    return zero_crossings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="ais file to filter",
    )

    args = parser.parse_args()

    if args.file:
        start_time = time.time()
        df = pd.read_csv(args.file,  engine="pyarrow")
        end_time = time.time()
        print("Read file:", (end_time - start_time), "seconds")
        print("Rows:", df.shape[0])
        zero_crossings = get_zero_crossings(df)
        with open("zeros.json", "w") as f:
            json.dump(zero_crossings, f)

    else:
        parser.print_usage()
        return sys.exit(1)


if __name__ == "__main__":
    main()
