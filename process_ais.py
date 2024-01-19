import json
import numpy as np
import pandas as pd
import argparse
import sys
import time
import collections
import simplekml
from polycircles import polycircles
import logging


BUFFER_SIZE = 12  # Generally each sample is 1m apart
MAX_SAMPLE_DELTA_SECONDS = 120  # Ensure buffer samples are within this delta
MIN_ALT_FEET = 10000  # Cutoff for AIS data
GPS_THRESHOLD_LOW = .3  # GPS loss is usually ~ 0
GPS_THRESHOLD_HIGH = 8  # GPS signal is usually > 7
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def get_zero_crossings(df):
    ''' Looks at all AIS data and runs the following algorithm: '''
    ''' Read the AIS file '''
    ''' 1. Add each flight hex to a dictionary / map '''
    ''' 2. For each hex store a buffer 12 samples (collections.deque) '''
    ''' 3. Add samples and if there’s a gap of more than MAX_SAMPLE_DELTA_SECONDS flush the buffer'''
    '''     a. This ensures each sample is within MAX_SAMPLE_DELTA_SECONDS of each other'''
    ''' 4. When the buffer is full look for a loss or recovery of signal between the first and last half of the buffer.'''
    '''     a. Since there are tons of signals eventually you’ll get a nice split in the center.'''
    '''     b. Loss of signal is when a half has an average NIC < GPS_THRESHOLD_LOW.'''
    '''     c. Signal is when a half has an average >= GPS_THRESHOLD_HIGH.'''
    '''     d. Looking at the data this is usually a complete loss or recovery from a complete loss. '''
    '''5.  When we find a loss or recovery, save the middle sample.'''

    def validate_sample_interval(dequeue_list, current_timestamp):
        '''Valid sample intervals are when buffer is empty or data is within MAX_SAMPLE_DELTA_SECONDS'''
        if len(dequeue_list) == 0:
            return True
        last_sample_timestamp = dequeue_list[-1].timestamp_u
        return (current_timestamp - last_sample_timestamp).total_seconds() < MAX_SAMPLE_DELTA_SECONDS

    def detect_loss_or_recovery(buffer):
        '''Divide buffer in half and look for loss or recovery of a signal'''
        prior_window = list(buffer)[:int(BUFFER_SIZE/2)]
        later_window = list(buffer)[int(BUFFER_SIZE/2):]
        # Get the average of both halfs
        prior_window_avg = sum(prior_window) / len(prior_window)
        later_window_avg = sum(later_window) / len(later_window)

        # Gain of GPS is when prior window was below and later is above
        recovery_of_gps = prior_window_avg < GPS_THRESHOLD_LOW and later_window_avg >= GPS_THRESHOLD_HIGH
        # Loss of GPS is when prior window was above and later is below
        loss_of_gps = prior_window_avg >= GPS_THRESHOLD_HIGH and later_window_avg < GPS_THRESHOLD_LOW
        return recovery_of_gps or loss_of_gps

    def get_new_buffer():
        return collections.deque(
            maxlen=BUFFER_SIZE)

    def get_nics_from_buffer(buffer):
        return [elem.nic for elem in buffer]

    start_time = time.time()
    flight_dict = {}
    zero_crossings = []

    for row in df.itertuples():
        try:
            hex_code = row.hex

            # Skip aircraft on the ground
            altitude = row.alt_baro
            if not str(altitude).isdigit() or int(altitude) < MIN_ALT_FEET:
                continue

            # If first time seen, add flight and initialize deque
            if hex_code not in flight_dict:
                flight_dict[hex_code] = get_new_buffer()

            # When too much time has past, flush the buffers
            if hex_code in flight_dict and not validate_sample_interval(flight_dict[hex_code], row.timestamp_u):
                flight_dict[hex_code] = get_new_buffer()

            # Add new record row
            flight_dict[hex_code].append(row)

            # If buffer is full start checking for zero crossings
            if hex_code in flight_dict and len(flight_dict[hex_code]) == BUFFER_SIZE:
                try:
                    nics = get_nics_from_buffer(flight_dict[hex_code])
                    # Trigger a zero crossing when gain or loss occurs
                    if detect_loss_or_recovery(nics):
                        zero_cross_record = flight_dict[hex_code][int(
                            BUFFER_SIZE/2)]
                        zero_crossing = {
                            "hex_code": zero_cross_record.hex,
                            "alt_baro": int(zero_cross_record.alt_baro),
                            "row_index": zero_cross_record.Index,
                            "lat": zero_cross_record.lat,
                            "lon": zero_cross_record.lon,
                            "nics": nics
                        }

                        zero_crossings.append(zero_crossing)

                        # Flush buffer when we find a zero crossing so we don't double count
                        flight_dict[hex_code] = get_new_buffer()
                except Exception as e:
                    logging.exception(
                        f"Zero crossing failure: {e}", exc_info=True)
        except Exception as e:
            logging.exception(
                f"Processing records failure: {e}", exc_info=True)

    end_time = time.time()
    logging.info(f"Processed file: {(end_time - start_time)} seconds")
    logging.info(f"Found {len(zero_crossings)} crossings")
    return zero_crossings


def horizontal_range(height):
    '''Get horizontal range to estimate where jammer is'''
    '''This should work because either the jammer just turned on, or just became visible'''
    '''Since the jammer has been on for weeks it's probably when it becomes visible'''
    RADIUS_EARTH = 6371000  # Radius in meters
    FEET_TO_METERS = 0.3048
    height *= FEET_TO_METERS
    return np.sqrt(2*RADIUS_EARTH*height + np.power(height, 2))


def create_kml(zero_crossings, filename='data.kml'):
    '''Create a kml file of all of the horizontal ranges as circles centered at loss or recovery of signal'''
    NUMBER_OF_VERTICES = 36  # 36 is best practice per developer
    kml = simplekml.Kml()
    for elem in zero_crossings:
        radius = horizontal_range(elem['alt_baro'])
        polycircle = polycircles.Polycircle(latitude=elem['lat'],
                                            longitude=elem['lon'],
                                            radius=radius,
                                            number_of_vertices=NUMBER_OF_VERTICES)
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
        required=True
    )

    args = parser.parse_args()

    if args.file:
        start_time = time.time()
        df = pd.read_csv(args.file,  engine="pyarrow")
        end_time = time.time()
        logging.info(f"Read file: {(end_time - start_time)} seconds")
        logging.info(f"Rows: {df.shape[0]} crossings")
        zero_crossings = get_zero_crossings(df)
        create_kml(zero_crossings)
        with open("zeros.json", "w") as f:
            json.dump(zero_crossings, f)

    else:
        parser.print_usage()
        return sys.exit(1)


if __name__ == "__main__":
    main()
