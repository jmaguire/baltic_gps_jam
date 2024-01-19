import json
import numpy as np
import pandas as pd
import argparse
import sys
import time
import collections
import simplekml
from polycircles import polycircles

BUFFER_SIZE = 30  # use even number for simple dividing in half
THRESHOLD_LOW = .5
THRESHOLD_HIGH = 8
RADIUS_EARTH = 6371000
FEET_TO_METERS = 0.3048
MAX_SAMPLE_DELTA_SECONDS = 120


# Compute zero crossings from a file of ais data
def get_zero_crossings(df):

    start_time = time.time()
    flight_dict = {}
    zero_crossings = []

    for index, row in df.iterrows():
        try:
            hex_code = row["hex"]

            # Altitude is not an int when on the ground
            try:
                altitude = int(row["alt_baro"])
            except ValueError:
                continue

            # Skip aircraft on the ground
            if altitude < 10000:
                continue

            # If first time seen, add buffer
            if hex_code not in flight_dict:
                flight_dict[hex_code] = {}
                flight_dict[hex_code]["nics"] = collections.deque(
                    maxlen=BUFFER_SIZE)
                flight_dict[hex_code]["index"] = collections.deque(
                    maxlen=BUFFER_SIZE)
                flight_dict[hex_code]["last_time"] = None

            # When too much time has past, flush the buffers
            if flight_dict[hex_code]["last_time"] and (row["timestamp_u"] - flight_dict[hex_code]["last_time"]).total_seconds() > MAX_SAMPLE_DELTA_SECONDS:
                flight_dict[hex_code]["nics"] = collections.deque(
                    maxlen=BUFFER_SIZE)
                flight_dict[hex_code]["index"] = collections.deque(
                    maxlen=BUFFER_SIZE)

            # Add row
            flight_dict[hex_code]["last_time"] = row["timestamp_u"]
            flight_dict[hex_code]["nics"].append(row["nic"])
            flight_dict[hex_code]["index"].append(index)

            # If buffer is full start checking for zero crossings
            if hex_code in flight_dict and len(flight_dict[hex_code]["nics"]) == BUFFER_SIZE:
                # Get the index halfway through the buffer
                # And divide the buffer in two
                split_index = int(BUFFER_SIZE/2)
                prior_window = list(flight_dict[hex_code]["nics"])[
                    :split_index]
                later_window = list(flight_dict[hex_code]["nics"])[
                    split_index:]

                # Get the average of both halfs
                prior_window_avg = sum(prior_window) / len(prior_window)
                later_window_avg = sum(later_window) / len(later_window)

                # Gain of GPS is when prior window was below and later is above
                gain_of_gps = prior_window_avg < THRESHOLD_LOW and later_window_avg >= THRESHOLD_HIGH

                # Loss of GPS is when prior window was above and later is below
                loss_of_gps = prior_window_avg >= THRESHOLD_HIGH and later_window_avg < THRESHOLD_LOW

                # Trigger a zero crossing when gain or loss occurs
                if gain_of_gps or loss_of_gps:
                    zero_cross_index = flight_dict[hex_code]["index"][split_index]
                    zero_cross_record = df.iloc[zero_cross_index]
                    zero_crossing = {
                        "hex_code": zero_cross_record["hex"],
                        "alt_baro": int(zero_cross_record["alt_baro"]),
                        "row_index": zero_cross_index,
                        "lat": zero_cross_record["lat"],
                        "lon": zero_cross_record["lon"],
                        "prior_nic_window": prior_window_avg,
                        "later_nic_window": later_window_avg,
                        "window": list(flight_dict[hex_code]["nics"]),
                    }
                    zero_crossings.append(zero_crossing)

                    # Flush buffer when we find a zero crossing so we don't double count
                    flight_dict[hex_code]["nics"] = collections.deque(
                        maxlen=BUFFER_SIZE)
                    flight_dict[hex_code]["index"] = collections.deque(
                        maxlen=BUFFER_SIZE)
                    flight_dict[hex_code]["last_time"] = None

        except Exception as e:
            print("I broke!")
            print(e)
            print(row)
            print(flight_dict[hex_code])
            return

    end_time = time.time()
    print("Processed file:", (end_time - start_time), "seconds")
    print("Found", len(zero_crossings), "crossings")
    return zero_crossings


def horizontal_range(height):
    height *= FEET_TO_METERS
    return np.sqrt(2*RADIUS_EARTH*height + np.power(height, 2))


def create_kml(zero_crossings, filename='data.kml'):
    kml = simplekml.Kml()
    for elem in zero_crossings:
        radius = horizontal_range(elem['alt_baro'])
        polycircle = polycircles.Polycircle(latitude=elem['lat'],
                                            longitude=elem['lon'],
                                            radius=radius,
                                            number_of_vertices=36)
        pol = kml.newpolygon(name="Columbus Circle, Manhattan",
                             outerboundaryis=polycircle.to_kml())
        pol.style.polystyle.color = \
            simplekml.Color.changealphaint(10, simplekml.Color.red)
    # Save to a file
    kml.save(filename)


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
        create_kml(zero_crossings)
        with open("zeros.json", "w") as f:
            json.dump(zero_crossings, f)

    else:
        parser.print_usage()
        return sys.exit(1)


if __name__ == "__main__":
    main()
