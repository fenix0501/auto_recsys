"""Input and time contracts for the aggregated KION user-item table."""
import pandas as pd


def historical_events(events,cutoff):
    """Filter first: future rows must not affect validation of historical inputs."""
    history=events[events['last_watch_dt']<cutoff].copy()
    if history[['user_id','item_id','last_watch_dt']].isna().any().any():
        raise ValueError('Historical keys and timestamps must be nonmissing')
    if history.duplicated(['user_id','item_id']).any():
        raise ValueError('KION contract requires one aggregated row per user-item in history')
    return history


def validate_metadata(users,items):
    for frame,key in [(users,'user_id'),(items,'item_id')]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f'Metadata must have one nonmissing unique {key} per row')


def days_elapsed(delta):
    return delta.dt.total_seconds()/86400
