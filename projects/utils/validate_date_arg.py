#!/usr/bin/python
import re


def date_arg(value):
    """
    Validate the --dates argument
    Function will ensure that the --dates argument matches a given regex.
    Possible values of the date are:
        - YYYYMMDD (eg. 20120515)
        - YYYYMMDD-YYYYMMDD (eg. 20140115-20140315)
        - yesterday
        - today
    Parameters
    ----------
        - value - *string* - automatically passed by the ArgumentParser object
    Returns
    -------
        Returns the value back to the ArgumentParser object
    Exceptions
    ----------
        - argparse.ArgumentTypeError - if the passed argument doesn't match the
    regex
    """
    # Regex check
    if not re.match('^(\d{8}|\d{8}-\d{8}|yesterday|today)$', value):
        raise argparse.ArgumentTypeError("must be in the form MMDD, "
                                         "MMDD-MMDD, yesterday, "
                                         "or today".format(value))
    # Make sure end date is >= start date
    if re.match("\d{8}-\d{8}", value):
        start, end = value.split('-')
        if not end >= start:
            raise argparse.ArgumentTypeError("The start date is less than the "
                                             "end date")
    return value