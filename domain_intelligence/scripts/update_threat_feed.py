"""Manually refresh the OpenPhish community feed and export training rows."""
from src.threat_feeds import update_openphish_feed, export_training_rows

if __name__ == "__main__":
    print(update_openphish_feed())
    print(export_training_rows())
