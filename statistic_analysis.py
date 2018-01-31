import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.stats import pearsonr


def string_simplify(raw_string):
    """
    A function converting methods' names into a simplified form.

    :param raw_string:          Method's names in original form
    :return:                    Simplified names
    """
    string_list = re.split('#', raw_string)
    if raw_string[0] == '#':
        if len(string_list) > 2:
            temp_string = '#' + string_list[1] + '#' + string_list[2]
        else:
            temp_string = '#' + string_list[1]
    else:
        if len(string_list) > 1:
            temp_string = string_list[0] + '#' + string_list[1]
        else:
            temp_string = string_list[0]
    return re.split("[:.|= ([]", temp_string)[0]


def jsonstring_to_dfdict(json_string, para_list):
    """
    A function extracting data and converting json string into a dict form.

    :param json_string:         Original json string in a line
    :param para_list:           Only parameters in the list, uuid, total cost and methods' names
                                                                        will be extracted to construct DataFrames
    :return:                    These dicts grouped into a list can be used to construct a DataFrame
    """
    total_dict = json.loads(json_string)
    df_dict = {"uuid": total_dict["uuid"], "total": total_dict["total"]}
    rawdata_dict = json.loads(total_dict["rawdata"])
    for para in para_list:
        df_dict[para] = rawdata_dict[para] if para in rawdata_dict else total_dict[para]
    jmt_dict = rawdata_dict["values"]["extVal"]["JMTReport"]
    for key in jmt_dict:
        df_dict[string_simplify(key)] = 1
    return df_dict


def read_jsonfiles(dir_path, file_list, para_list):
    """
    A function analyzing JSON files into a raw dataframe.

    :param dir_path:          Path of the JSON directory
    :param file_list:         Determine which file in the dir will be read. If empty, read all files
    :param para_list:         Only parameters in the list will be extracted to construct DataFrames
    :return:                  Raw dataframe constructed from the JSON files
    """
    df_list = []
    if not file_list:
        file_list = os.listdir(dir_path)
    for filename in file_list:
        try:
            logging.info("Reading file \"{}\"...\n".format(filename))
            with open(dir_path + filename, encoding="utf-8") as jsonfile:
                string_list = jsonfile.readlines()
                for string in string_list:
                    df_dict = jsonstring_to_dfdict(string, para_list)
                    df_list.append(df_dict)
        except Exception as e:
            logging.warning(e)
            logging.warning("File \"{}\" is not a format-valid JSON file.\n".format(filename))
    df = pd.DataFrame(df_list)
    return df


def pretreat_dataframe(df, filter_dict):
    """
    A function transform the raw dataframe according to the requirement of filter,
                                                        Adjust the order of columns and Fill NaN with zero.

    :param df:              Input raw dataframe
    :param filter_dict:     Rule of filtering
    :return:                Treated dataframe
    """
    # Filter DataFrame
    if filter_dict:
        for key, value in filter_dict.items():
            df = df[df[key].isin(value)]
    # Adjust the order of columns. Make id be the first, total the last. FillNan.
    df_id = df["uuid"]
    df_total = df["total"]
    df = df.drop(labels=["uuid", "total"], axis=1)
    df.insert(0, "uuid", df_id)
    df.insert(len(df.columns), "total", df_total)
    return df.fillna(0)


def extract_arrays(df, para_list):
    """
    A function returning x, y in the form of arrays and columns (list of methods' name).

    :param df:              The whole dataframe built from JSON file
    :param para_list:       Parameters used to get sub DataFrames in different conditions
    :return:                x, y and columns
    """
    df_x = df[[item for item in df.columns if item not in para_list and item != "total" and item != "uuid"]]
    x_array = df_x.values
    y_array = df["total"].values
    name_list = df_x.columns.tolist()
    return x_array, y_array, name_list


def correlation_analysis(x, y):
    """
    A function used for correlation analysis between x (methods) and y (total cost).
                                                            Invalid value will be converted to -2.

    :param x:               Array x
    :param y:               Array y
    :return:                List of correlation coefficients
    """
    c_list = []
    np.seterr(invalid="raise")
    for column in range(x.shape[1]):
        try:
            c_list.append(pearsonr(x[:, column], y)[0])
        except FloatingPointError:
            c_list.append(np.nan)
    return c_list


def rf_analysis(x, y):
    """
    A function used to model x (methods) and y (total cost) using Random Forests.
                                                            Output weights of methods in a list.

    :param x:               Array x
    :param y:               Array y
    :return:                List of weights
    """
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(x, y)
    w_list = clf.feature_importances_.tolist()
    return w_list


def build_result(name_list, estimation_list, estimator_name):
    """
    A function used to build the final analysis result in dataframe form.

    :param name_list:               List of methods' names
    :param estimation_list:         List of estimation values
    :param estimator_name:          Name of estimator, e.g., "weights", "corr",
    :return:                        Dataframe of analysis result including 2 columns: name, estimation values
    """
    df_dict = {"methodName": name_list, estimator_name: estimation_list}
    df = pd.DataFrame(df_dict, columns=["methodName", estimator_name])
    sorted_df = df.sort_values(by=estimator_name, ascending=False).reset_index(drop=True)
    return sorted_df


def df_visualize(df, nums_plot):
    """
    A function visualizing dataframe of analysis result.

    :param df:                      Dataframe of analysis result with two columns (method's name, value).
    :param nums_plot:               Data of Top nums_plot methods will be visualized
    :return:                        Handle of figure
    """
    plt.figure(figsize=(nums_plot, 8))
    for index, row in df.iloc[:nums_plot].iterrows():
        plt.bar(index + 1, row.iloc[1], width=0.5, label=row.iloc[0])
    plt.xticks(tuple(range(1, nums_plot + 1)))
    plt.xlabel("Rank of method", fontsize=12)
    plt.ylabel(df.columns[1], fontsize=12)
    ax = plt.gca()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig = plt.gcf()
    return fig


def config_logger(output_path):
    """
    A function to simply configure the logger.

    :param output_path:             Output path of the log file
    :return:                        Nothing
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt="%d %b %Y %H:%M:%S",
                        filename=output_path + "statistic_analysis.log",
                        filemode="w")


def main():
    #  Load configuration
    dir_path = CONFIG["DIR_PATH"]
    file_list = CONFIG["FILE_LIST"]
    para_list = CONFIG["PARA_LIST"]
    filter_dict = CONFIG["FILTER_DICT"]
    output_path = CONFIG["OUTPUT_PATH"]
    nums_plot = CONFIG["NUMS_PLOT"]
    fig_dpi = CONFIG["DPI"]

    #  Config logger
    config_logger(output_path)

    #  Read json files
    df_raw = read_jsonfiles(dir_path, file_list, para_list)
    if df_raw.empty:
        logging.error("No valid JSON file in directory \"{}\".\n".format(dir_path))
        raise ImportError("No valid JSON file in directory \"{}\".".format(dir_path))
    logging.info("Json files have been read.\n")

    #  Preprocess dataframe
    df_data = pretreat_dataframe(df_raw, filter_dict)
    logging.info("Dataframe has been preprocessed.\n")

    #  Extract arrays x and y
    [x_array, y_array, name_list] = extract_arrays(df_data, para_list)
    logging.info("Arrays x and y have been extracted.\n")

    #  Statistical analysis
    list_cc = correlation_analysis(x_array, y_array)
    logging.info("Correlation analysis finished.\n")
    list_rf = rf_analysis(x_array, y_array)
    logging.info("Random Forests analysis finished.\n")

    #  Build results
    result_cc = build_result(name_list, list_cc, "corr")
    result_rf = build_result(name_list, list_rf, "weights")
    logging.info("Results have been built.\n")

    #  Output analysis results into csv files
    result_cc.to_csv(output_path + "result_cc.csv", index=False)
    result_rf.to_csv(output_path + "result_rf.csv", index=False)
    logging.info("csv files have been exported.\n")

    #  Output analysis results into figures
    df_visualize(result_cc, nums_plot).savefig(output_path + "result_cc.png", dpi=fig_dpi)
    df_visualize(result_rf, nums_plot).savefig(output_path + "result_rf.png", dpi=fig_dpi)
    logging.info("Figures have been exported.\n")
    df_visualize(result_cc, nums_plot).show()
    df_visualize(result_rf, nums_plot).show()


if __name__ == "__main__":
    CONFIG = {
        "DIR_PATH": "/Users/hajiang2/Documents/jsonfiles.json/",
        "FILE_LIST": ['1'],
        "PARA_LIST": ["cmrflag", "cmrversion", "pmrflag", "appType", "browser", "jointype", "os", "serverId", "siteId",
                      "userid", "usertype", "version"],
        "FILTER_DICT": {},
        # "FILTER_DICT": {"cmrflag": ["true"], "jointype": ["UpgradeUser", "NewUser", "ReturnUser"]},
        "OUTPUT_PATH": "/Users/hajiang2/Documents/analysis_result/",
        "NUMS_PLOT": 20,
        "DPI": 500
    }
    main()
