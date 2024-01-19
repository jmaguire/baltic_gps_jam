import json
import numpy as np
import pandas as pd
import argparse
import sys
import time
import collections
import simplekml
from polycircles import polycircles

BUFFER_SIZE = 12  # use even number for simple dividing in half
THRESHOLD_LOW = .3
THRESHOLD_HIGH = 8
RADIUS_EARTH = 6371000
FEET_TO_METERS = 0.3048
MAX_SAMPLE_DELTA_SECONDS = 120
MIN_ALT_FEET = 10000


# Compute zero crossings from a file of ais data
def get_zero_crossings(df):

    def validate_sample_interval(dequeue_list, record):
        # If there are no samples, this is valid!
        if len(dequeue_list) == 0:
            return True
        last_sample_timestamp = dequeue_list[-1]["timestamp"]
        return (record["timestamp_u"] - last_sample_timestamp).total_seconds() < MAX_SAMPLE_DELTA_SECONDS

    def detect_loss_or_recovery(buffer):
        prior_window = list(buffer)[:int(BUFFER_SIZE/2)]
        later_window = list(buffer)[int(BUFFER_SIZE/2):]
        # Get the average of both halfs
        prior_window_avg = sum(prior_window) / len(prior_window)
        later_window_avg = sum(later_window) / len(later_window)

        # Gain of GPS is when prior window was below and later is above
        gain_of_gps = prior_window_avg < THRESHOLD_LOW and later_window_avg >= THRESHOLD_HIGH
        # Loss of GPS is when prior window was above and later is below
        loss_of_gps = prior_window_avg >= THRESHOLD_HIGH and later_window_avg < THRESHOLD_LOW
        return gain_of_gps or loss_of_gps

    def get_new_buffer():
        return collections.deque(
            maxlen=BUFFER_SIZE)

    def get_elems_from_buffer(buffer, key):
        return [elem[key] for elem in buffer]

    start_time = time.time()
    flight_dict = {}
    zero_crossings = []

    for index, row in df.iterrows():
        try:
            hex_code = row["hex"]

            # Skip aircraft on the ground
            altitude = row["alt_baro"]
            if not str(altitude).isdigit() or int(altitude) < MIN_ALT_FEET:
                continue

            # If first time seen, add flight and initialize deque
            if hex_code not in flight_dict:
                flight_dict[hex_code] = get_new_buffer()

            # When too much time has past, flush the buffers
            if hex_code in flight_dict and not validate_sample_interval(flight_dict[hex_code], row):
                flight_dict[hex_code] = get_new_buffer()

            # Add new record row
            flight_dict[hex_code].append({
                "nic": row["nic"], "timestamp": row["timestamp_u"], "index": index
            })

            # If buffer is full start checking for zero crossings
            if hex_code in flight_dict and len(flight_dict[hex_code]) == BUFFER_SIZE:
                try:
                    nics = get_elems_from_buffer(flight_dict[hex_code], "nic")
                    # Trigger a zero crossing when gain or loss occurs
                    if detect_loss_or_recovery(nics):
                        zero_cross_index = flight_dict[hex_code][int(
                            BUFFER_SIZE/2)]["index"]
                        zero_cross_record = df.iloc[zero_cross_index]
                        zero_crossing = {
                            "hex_code": zero_cross_record["hex"],
                            "alt_baro": int(zero_cross_record["alt_baro"]),
                            "row_index": zero_cross_index,
                            "lat": zero_cross_record["lat"],
                            "lon": zero_cross_record["lon"],
                            "nics": nics
                        }

                        # timestamps = [str(elem) for elem in get_elems_from_buffer(
                        #     flight_dict[hex_code], "timestamp")]
                        # zero_crossing[timestamps] = timestamps

                        zero_crossings.append(zero_crossing)

                        # Flush buffer when we find a zero crossing so we don't double count
                        flight_dict[hex_code] = get_new_buffer()
                except Exception as e:
                    print("zero crossing broke!")
                    print(e)
                    print(row)
                    return

        except Exception as e:
            print("main loop!")
            print(e)
            print(row)
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
        pol = kml.newpolygon(name="KRAZY KAT BAT KILLER",
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
