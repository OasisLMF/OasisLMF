#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import argparse
import os
from subprocess import Popen

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load model data')
    parser.add_argument(
        '-s', '--sql_server_ip', type=str,  required=True,
        help="The SQL Server IP.")
    parser.add_argument(
        '-n', '--environment_name', type=str,  required=True,
        help="The environment name.")
    parser.add_argument(
        '-l', '--login_password', type=str,  required=True,
        help='The SQL login password.')
    parser.add_argument(
        '-a', '--keys_service_ip', type=str,  required=True,
        help='The Keys Server API IP.')
    parser.add_argument(
        '-A', '--keys_service_port', type=str,  required=True,
        help='The Keys Server API port.')
    parser.add_argument(
        '-o', '--oasis_api_ip', type=str,  required=True,
        help='The Oasis API IP.')
    parser.add_argument(
        '-O', '--oasis_api_port', type=str,  required=True,
        help='The Oasis API PORT.')

    args = parser.parse_args()

    sql_server_ip = args.sql_server_ip
    environment_name = args.environment_name
    login_password =  args.login_password
    keys_service_ip = args.keys_service_ip
    keys_service_port = args.keys_service_port
    oasis_api_ip = args.oasis_api_ip
    oasis_api_port = args.oasis_api_port

    def run_command(desc, cmd, exit_on_fail=True, retry=False):
        print cmd
        proc = Popen(cmd, shell=True)
        proc.wait()
        if (proc.returncode > 0 and exit_on_fail):
            print ("FAIL: {}".format(desc))
            if retry:
                print("RETRY: {}".format(desc))
                run_command(desc, cmd, True, False)
            else:
                exit(255)

    sql_script = "LoadModelData_run.sql"
    shutil.copyfile("LoadModelData.sql", sql_script)

    run_command(
        "Update build script with keys service IP",
        "sed -i -e s/%KEYS_SERVICE_IP%/{}/g {}".format(keys_service_ip, sql_script))
    run_command(
        "Update build script with keys service port",
        "sed -i -e s/%KEYS_SERVICE_PORT%/{}/g {}".format(keys_service_port, sql_script))
    run_command(
        "Update build script with Oasis API IP",
        "sed -i -e s/%OASIS_API_IP%/{}/g {}".format(oasis_api_ip, sql_script))
    run_command(
        "Update build script with Oasis API port",
        "sed -i -e s/%OASIS_API_PORT%/{}/g {}".format(oasis_api_port, sql_script))

    run_command(
        "Creating db",
        "sqlcmd -S {} -d Flamingo_{} -U {} -P {} -i {}".format(
            sql_server_ip, environment_name, environment_name, login_password, sql_script))

    os.remove(sql_script)