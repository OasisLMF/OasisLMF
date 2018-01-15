# -*- coding: utf-8 -*-

"""
Utils for running SQL commands against a database.
"""

__all__ = [
    'bcp',
    'check_connection',
    'CONN_STRING',
    'DB_CONFIG',
    'execute',
    'fetch_one',
    'fetchall',
    'read_db_config'
]

import logging
import os

import pyodbc

from .log import oasis_log

DB_CONFIG = {}
CONN_STRING = ""


def bcp(table, outfile):
    """
    BCP a table to a file.
    """
    command = "freebcp {}.dbo.{} out {} -c -t, -U {} -P {} -S {}:{}".format(
        DB_CONFIG['database'], table, outfile,
        DB_CONFIG['username'], DB_CONFIG['password'],
        DB_CONFIG['server'], DB_CONFIG['port'])
    os.system(command)


def read_db_config(config_parser):
    """
    Read an Oasis standard db config
    """

    global DB_CONFIG, CONN_STRING

    DB_CONFIG['server'] = config_parser['FLAMINGO_DB_IP']
    DB_CONFIG['port'] = config_parser['FLAMINGO_DB_PORT']
    DB_CONFIG['username'] = config_parser['FLAMINGO_DB_USERNAME']
    DB_CONFIG['password'] = config_parser['FLAMINGO_DB_PASSWORD']
    DB_CONFIG['database'] = config_parser['FLAMINGO_DB_NAME']

    CONN_STRING = "DRIVER={};PORT={};SERVER={};DATABASE={};uid={};pwd={}".format(
        '{FreeTDS}',
        DB_CONFIG['port'], DB_CONFIG['server'], DB_CONFIG['database'],
        DB_CONFIG['username'], DB_CONFIG['password'])


@oasis_log()
def execute(sql, *parameters):
    """
    Execute a SQL statement with specified parameters.
    """
    conn = pyodbc.connect(CONN_STRING)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(sql, parameters)
    conn.commit()


@oasis_log()
def fetch_one(sql, *parameters):
    """
    Execute a SQL statement with specified parameters, and return a
    single row.
    """
    conn = pyodbc.connect(CONN_STRING)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(sql, parameters)
    row = cursor.fetchone()
    conn.commit()
    return row


@oasis_log()
def fetchall(sql, *parameters):
    """
    Execute a SQL statement with specified parameters, and return
    all rows.
    """
    conn = pyodbc.connect(CONN_STRING)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(sql, parameters)
    rows = cursor.fetchall()
    logging.getLogger().info("Feteched {} rows".format(len(rows)))
    conn.commit()
    return rows


@oasis_log()
def check_connection():
    """
    Run a simple query against the Flamingo database
    """
    try:
        db_summary = "PORT={};SERVER={};DATABASE={}".format(
            DB_CONFIG['port'], DB_CONFIG['server'], DB_CONFIG['database'])
        logging.getLogger().info(
            "Checking connection: {}".format(db_summary))
        fetchall("SELECT * FROM version")
    except Exception as e:
        logging.getLogger().error(
            "Failed to connect to database: {}".format(db_summary))
        return False
    return True
