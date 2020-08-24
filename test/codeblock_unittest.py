import unittest
import pandas as pd
import numpy as np
import seaborn as sns
import json
import re
import ast
from sklearn import datasets
from codeblock_class import CodeBlock

class TestCrowdBlocks(unittest.TestCase):
    iris = datasets.load_iris()
    df1 = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])
    df_1 = pd.DataFrame(np.random.randint(low=1, high=10, size=(20,3)), columns=['a', 'b', 'c'])
    df2 = pd.DataFrame({'a': [3, 4, 5], 'b': [0, 1, 2], 'c': [6, 'z', 8]})
    df3 = pd.DataFrame({'a': [3, 4, 5], 'b': [0, 1, 2], 'c': [6, np.nan, 8]})
    df4 = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/crop_yield.csv")
    df5 = pd.DataFrame({'a': ["X", "Y", "Z"], 'b': ["T", "U", "V"]})
    df6 = pd.DataFrame({'a': [3, 4, 5], 'b': [0, 1, 2], 'c': [np.nan, np.nan, 8]})
    df7 = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12], [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
    columns=['Apple', 'Orange', 'Banana', 'Pear'], index=['Basket1', 'Basket2', 'Basket3', 'Basket4', 'Basket5', 'Basket6'])
    df8 = pd.DataFrame({'a': [True, False]*3, 'b': ['' ,'']*3, 'c': ['1.0', '2.0']*3})
    df9 = pd.DataFrame({'a': ['','']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})
    df10 = None
    df11 = pd.DataFrame({'a': ['','']*3, 'b': [3, '']*3 , 'c': ['1.0', '2.0']*3})
    df12 = pd.DataFrame({'a': [np.nan, np.nan]*3, 'b': [np.nan, np.nan]*3, 'c': [np.nan, np.nan]*3})
    df13= pd.DataFrame({'a': ['10','1']*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})
    df14 = pd.DataFrame({'a': [10,1]*3, 'b': [3, 3]*3, 'c': ['1.0', '2.0']*3})
    df15 = pd.DataFrame({'a': [10], 'b': [3], 'c': [1]})
    df16 = pd.DataFrame({'a': [3, 4, 5]*4, 'b': [0, 1, 2]*4, 'c': [np.nan, np.nan, 8]*4})
    c = CodeBlock()

# NaN strategies
# If there are only a handful of rows that contain nans, then drop the entire row.
# (as long as we have at least 3 rows remaining with no nan values)
# Similar strategy for columns, but then have to determine if it's still a viable analysis without that col
# If a column is majority nan values, then drop the column. If a column isnt majory nan, then drop by rows
# make this into a bug/ issue
# for current nan tests, write them according to our current expected behavior (i.e. returns empty df)

# focus on this later:
# ["mean.py", "top5categories.py" ... ]
# fold (test1(x)) [] lst

# don't focus on tests passing; focus on finding issues in the codeblocks (which there probably will be)

############################ Helper Functions ##################################

    # strip and reformat res['result'] into a dataframe
    def converter(self, obj):
        # gives back a string representation of the data
        data = obj.partition("data\":")[2]

        # get row names
        indices = [m.start() for m in re.finditer('index', data)]
        lst = map(lambda x : x + 8, indices)
        row_names = map(lambda x : data[x:data.find('\"', x)], lst)

        # get data values
        raw_cols = re.findall('\{(.*?)\}', data)
        cols = map(lambda x : x[x.index(",") + 1:], raw_cols)
        cols = map(lambda x : "{" + x + "}", cols)
        cols_data = map(lambda x : ast.literal_eval(x), cols)

        # create the dataframe
        return pd.DataFrame(cols_data, row_names)

    # return True if a df has only numeric values and False otherwise
    def numeric_check(self, df):
        lst = df.apply(lambda s : pd.to_numeric(s, errors='coerce').notnull().all()).to_string()
        return False if "False" in lst else True


############################### Tests ##########################################
    # NOTE: Possible invalid dataframe inputs:
    # Nonnumeric values
    # Non categorical values
    # Null dataframe
    # Empty dataframe
    # Invalid datarame - i.e. columns have mixed data types

    ###################### ANOVA Tests ###########################
    def test_anova(self):
        res = self.c.anova_variance(self.df4, [], "", "")
        self.assertTrue(res['result'])

    def test_anova_nonnumeric(self):
        res = self.c.anova_variance(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_anova_noncategorical(self):
        res = self.c.anova_variance(self.df1, [], "", "")
        self.assertEqual(res['result'], "Dataframe contained incorrect values")

    # returns an error, but our error checking might be too strict
    # we don't want to throw out a whole analysis because of one nan value
    def test_anova_nan(self):
        res = self.c.anova_variance(self.df3, [], "", "")
        #print(res['result'])

    ###################### Bootstrap Tests ###########################
    def test_bootstrap(self):
        res = self.c.bootstrap(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_bootstrap_inner_mixed(self):
        res = self.c.bootstrap(self.df2, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_bootstrap_nonnumeric(self):
        res = self.c.bootstrap(self.df5, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    # why does this return "Dataframe has no numeric values?"
    # bootstrap should just select the columns with numeric data
    def test_bootstrap_nan(self):
        res = self.c.bootstrap(self.df6, [], "", "")
        #self.assertFalse(res['result'], "Dataframe has no numeric values")

    ###################### Category Boxplot Tests ###########################
    def test_category_boxplot(self):
        res = self.c.cat_boxplot(self.df4, [], "", "")
        self.assertTrue(res['result']) # this is an image, so for now just check if it runs and returns and image_list

    def test_category_boxplot_noncategorical(self):
        res = self.c.cat_boxplot(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_category_boxplot_nonnumeric(self):
        res = self.c.corr(self.df8, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)


    ###################### Category Count Tests ###########################
    def test_category_count(self):
        res = self.c.cat_count(self.df4, [], "", "")
        self.assertTrue(res['result'])

    def test_category_count_noncategorical(self):
        res = self.c.cat_count(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)


    ###################### Compute Covariance Matrix Tests ###########################
    # should this really give "matrix is singular" for df7?
    # res = self.c.compute_covariance_matrix(self.df7, [], "", "")
    # print(res['result'])

    def test_compute_covariance_matrix_nonnumeric(self):
        res = self.c.compute_covariance_matrix(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    # this throws a weird ValueError: shapes(3,) and (10,10) not aligned
    def test_compute_covariance_matrix_singular(self):
        res = self.c.compute_covariance_matrix(self.df1, [], "", "")
        self.assertEqual(res['result'], "Matrix is singular")


    ###################### CORRELATION Tests ###########################
    def test_corr(self):
        res = self.c.corr(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.size, 9)
        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.shape[1], 3)
        self.assertTrue(self.numeric_check(df))

    def test_corr_nonnumeric(self):
        res = self.c.corr(self.df8, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_corr_mixed(self):
        res = self.c.corr(self.df9, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_corr_inner_mixed(self):
        res = self.c.corr(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_corr_nan(self):
        res = self.c.corr(self.df12, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_corr_null(self):
        res = self.c.corr(self.df10, [], "", "")
        self.assertEqual(res['result'], "Null dataframe needs numeric values")

    ################## COMPUTE PERCENTILES RANGE Tests #######################
    def test_cpr(self):
        res = self.c.compute_percentiles_range(self.df1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_cpr_nonnumeric(self):
        res = self.c.compute_percentiles_range(self.df5, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_cpr_mixed(self):
        res = self.c.compute_percentiles_range(self.df9, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_cpr_inner_mixed(self):
        res = self.c.compute_percentiles_range(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    # currently, the function works on a df of only nan values, and returns a df with nan vals
    # is this what we want, or do we want a catch?
    def test_cpr_nan(self):
        res = self.c.compute_percentiles_range(self.df12, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe needs numeric values")

    def test_cpr_null(self):
        res = self.c.corr(self.df10, [], "", "")
        self.assertEqual(res['result'], "Null dataframe needs numeric values")

    ################### CORRELATION HEAT MAP Tests ############################
    def test_corr_heatmap(self):
        res = self.c.corr_heatmap(self.df1, [], "", "")
        self.assertNotIsInstance(res['output'], Exception)

    # NOTE: The block still ran without error
    # NOTE: Should there be an explicit check for this in the block?
    def test_corr_heatmap_nonnumeric(self):
        res = self.c.corr_heatmap(self.df8, [], "", "")


    def test_corr_heatmap_mixed(self):
        res = self.c.corr_heatmap(self.df9, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)


    # NOTE: This still ran without producing any error, this needs to be fixed
    def test_corr_heatmap_inner_mixed(self):
        res = self.c.corr_heatmap(self.df2, [], "", "")
        # self.assertEqual(res_test['result'], "Dataframe needs numeric values")

    def test_corr_heatmap_nan(self):
        res = self.c.corr_heatmap(self.df6, [], "", "")
        # self.assertEqual(res_nan['result'], "Dataframe needs numeric values")

    # NOTE: this will fail because corrheatmap doesn't do a null check
    def test_corr_heatmap_null(self):
        res = self.c.corr_heatmap(self.df10, [], "", "")
        self.assertEqual(res['result'], "Null Dataframe needs numeric values")

    ################### DECISION TREE CLASSIFIER Tests #############################
    # Tests behavior with mixed datatype columns (with >= 1 column of numeric datatype)
    # NOTE: Modified block to check for <= 1 quantitative columns
    # NOTE: Ran with 1 numeric column - threw shape error
    # NOTE: Ran with 2 numeric column - did not throw error - maybe we need to have atleast two numeric columns

    def test_dtc(self):
        res = self.c.decision_tree_classifier(self.df_1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_dtc_mixed_invalid(self):
        res = self.c.decision_tree_classifier(self.df13, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    def test_dtc_mixed_valid(self):
        res = self.c.decision_tree_classifier(self.df14, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_dtc_null(self):
        res = self.c.decision_tree_classifier(self.df10, [], "", "")
        self.assertEqual(res['result'], "Null Dataframe needs numeric values")

    ################### DECISION TREE REGRESSOR Tests #############################
    def test_dtr(self):
        res = self.c.decision_tree_regressor(self.df_1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_dtr_nan(self):
        res = self.c.decision_tree_regressor(self.df3, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    # this shouldn't fail. Somehow the is_numeric_dtype check fails on column 'a'
    # (which consists of bool values), so fit(x,y) is still called on the df, resulting in
    # ValueError: Found array with 0 feature(s) (shape=(6, 0)) while a minimum of 1 is required.
    def test_dtr_nonnumeric(self):
        res = self.c.decision_tree_regressor(self.df8, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    ###################### Demo Hstack Tests ##############################
    def test_demo_hstack_1(self):
        res = self.c.demo_hstack(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertNotIsInstance(df, Exception)

    def test_demo_hstack_2(self):
        res = self.c.demo_hstack(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertNotIsInstance(df, Exception)

    def test_demo_hstack_mixed(self):
        res = self.c.demo_hstack(self.df14, [], "", "")
        df = self.converter(res['result'])
        self.assertNotIsInstance(df, Exception)

    def test_demo_hstack_nonnumeric(self):
        res = self.c.demo_hstack(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    # NOTE: we do y1 = 1 / y in this method, which gives a runtime warning. We might want to check y != 0
    # Also, this test does assertNotEqual because the analysis should still run even when there is a nan value
    def test_demo_hstack_nan(self):
        res = self.c.demo_hstack(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe needs numeric values")

    ###################### Demo Log Space Tests ##############################
    def test_demo_log_space_1(self):
        res = self.c.demo_log_space(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_demo_log_space_2(self):
        res = self.c.demo_log_space(self.df_1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_demo_log_space_3(self):
        res = self.c.demo_log_space(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_demo_log_space_nonnumeric(self):
        res = self.c.demo_log_space(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_demo_log_space_nan(self):
        res = self.c.demo_log_space(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe needs numeric values")

    ###################### Demo Mat Show Tests ##############################
    def test_demo_mat_show_1(self):
        res = self.c.demo_mat_show(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_demo_mat_show_mixed(self):
        res = self.c.demo_mat_show(self.df14, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_demo_mat_show_nonnumeric(self):
        res = self.c.demo_mat_show(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    # running this test will also product a plot (as it should, which is another sanity check)
    def test_demo_mat_show_nan(self):
        res = self.c.demo_mat_show(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe needs numeric values")

    ################# Distribution Quantitative Category Tests #################
    # NOTE: What df would work for this? I thought df4 would work but maybe I'm not understanding the analysis correctly
    def test_dist_quant_category_1(self):
        res = self.c.dist_quant_category(self.df4, [], "", "")
        print(res['result'])

    def test_dist_quant_category_onlynumeric(self):
        res = self.c.dist_quant_category(self.df1, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric AND category values")

    def test_dist_quant_category_onlycategorical(self):
        res = self.c.dist_quant_category(self.df5, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric AND category values")

    ###################### Distribution Quantitative Tests ####################
    def test_dist_num_1(self):
        res = self.c.dist_num(self.df1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_dist_num_2(self):
        res = self.c.dist_num(self.df7, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_dist_num_nonnumeric(self):
        res = self.c.dist_num(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    # TODO: NaN check for dist_num
    def test_dist_num_nan(self):
        res = self.c.dist_num(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numerical values")

    ################## Distribution Quantitative Two Categories ###############
    def test_dist_two_categories_1(self):
        res = self.c.dist_two_categories(self.df4, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    # NOTE: This shouldn't fail. Somehow res is not being populated at all
    def test_dist_two_categories_noncategorical(self):
        res = self.c.dist_two_categories(self.df1, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no categorical values")

    ###################### Drop NaN Columns Tests ###########################
    def test_nan_cols_1(self):
        res = self.c.drop_cols(self.df3, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[1], 2)

    def test_nan_cols_2(self):
        res = self.c.drop_cols(self.df16, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    # returns the original df because no columns were dropped
    # the 'type' of the res is an error, but it returns what we want (the df)
    def test_nan_cols_nonan(self):
        res = self.c.drop_cols(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[1], 3)

    ###################### Drop NaN Rows Tests ###########################
    def test_nan_rows_1(self):
        res = self.c.drop_rows(self.df3, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_nan_rows_nonnan(self):
        res = self.c.drop_rows(self.df1, [], "", "")
        self.assertEqual(res['description'], "Dataframe has no rows with NaN entries")

    ###################### Eval Model Predictions Tests #########################
    def test_eval_model_1(self):
        res = self.c.eval_model_predictions(self.df1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_eval_model_nonnumeric(self):
        res = self.c.eval_model_predictions(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_eval_model_nan(self):
        res = self.c.eval_model_predictions(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Extra Trees Classifier Tests ########################
    # NOTE: this analysis runs fine, but there is a warning about the minimum number of members
    # allowed in a class. Do we care about this?
    def test_extra_trees_1(self):
        res = self.c.extra_trees_classifier(self.df_1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_extra_trees_nonnumeric(self):
        res = self.c.extra_trees_classifier(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    # NOTE: this shouldn't fail. ETC should be catching nan values
    def test_extra_trees_nan(self):
        res = self.c.extra_trees_classifier(self.df3, [], "", "")
        self.assertEqual(res['result'], "Dataframe needs numeric values")

    ###################### First 10 Tests ###########################
    def test_first_ten_1(self):
        res = self.c.firstTen(self.df4, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[0], 10)

    def test_first_ten_2(self):
        res = self.c.firstTen(self.df2, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.size, 9)

    ###################### Fit Decision Tree Tests ###########################
    # NOTE: what is the expected input and behavior of this function? FDT doesn't do any input validation
    def test_fit_decision_tree_1(self):
        res = self.c.decision_tree_regressor(self.df_1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_fit_decision_tree_nonnumeric(self):
        res = self.c.decision_tree_regressor(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    ###################### Descriptive Statistics Tests ########################
    def test_des(self):
        res = self.c.des(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_des_nan(self):
        res = self.c.des(self.df3, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    ###################### Matrix Norm Tests ###########################
    def test_matrix_norm_1(self):
        res = self.c.matrix_norm(self.df_1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_matrix_norm_nonnumeric(self):
        res = self.c.matrix_norm(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    def test_matrix_norm_nan(self):
        res = self.c.matrix_norm(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Mean Tests ###########################
    def test_mean(self):
        res = self.c.mean(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.size, 3)

    def test_mean_nonnumeric(self):
        res = self.c.mean(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    ###################### Numerical Boxplot Tests ###########################
    def test_num_boxplot_1(self):
        res = self.c.num_boxplot(self.df_1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_num_boxplot_nonnumeric(self):
        res = self.c.num_boxplot(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    def test_num_boxplot_nan(self):
        res = self.c.num_boxplot(self.df3, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Outer Join Tests ###########################
    def test_outer_join_1(self):
        res = self.c.outer_join(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    # TODO: add a null check to outer_join function
    def test_outer_join_null(self):
        res = self.c.outer_join(self.df10, [], "", "")

    def test_outer_join_nan(self):
        res = self.c.outer_join(self.df6, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    ###################### Plot via Limit Tests ###########################
    # NOTE: when does this function work? Every df I try gives IndexError:
    # list index is out of range. So this test fails
    def test_plot_via_limit(self):
        res = self.c.plot_via_limit(self.df2, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    def test_plot_via_limit_nonnumeric(self):
        res = self.c.plot_via_limit(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    # tests the case of <2 numeric columns
    def test_plot_via_limit_invalid(self):
        res = self.c.plot_via_limit(self.df9, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    ########################## Plot Tests #################################
    def test_plot(self):
        res = self.c.plot(self.df7, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    def test_plot_nonnumeric(self):
        res = self.c.plot(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    ########################## Predict Tests ###############################
    # What is the expected input for predict_test?

    # tests case of <2 rows
    def test_predict_invalid(self):
        res = self.c.predict_test(self.df15, [], "", "")
        self.assertEqual(res['result'], "Dataframe has less than two rows")

    # TODO: add a numeric data check to predict_test
    # this test will error because we currently do not validate the input type
    def test_predict_nonnumeric(self):
        res = self.c.predict_test(self.df11, [], "", "")

    ###################### Probability Density Plot Tests ######################
    def test_probability_density_plot_1(self):
        res = self.c.probability_density_plot(self.df_1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)


    def test_probability_density_plot_nonnumeric(self):
        res = self.c.probability_density_plot(self.df11, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    def test_probability_density_plot_nan(self):
        res = self.c.probability_density_plot(self.df6, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    ###################### Quantitative Bar Plot Tests ######################
    def test_quant_bar_plot_1(self):
        res = self.c.quantitative_bar_plot(self.df_1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    def test_quant_bar_plot_nonnumeric(self):
        res = self.c.quantitative_bar_plot(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    def test_quant_bar_plot_nan(self):
        res = self.c.quantitative_bar_plot(self.df6, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Random Forest Classifier Tests ######################
    # this analysis runs and produces a df, but we get a split error
    def test_random_forest_1(self):
        res = self.c.random_forest_classifier(self.df_1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)

    def test_random_forest_nonnumeric(self):
        res = self.c.random_forest_classifier(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    # TODO: handle NaN values in random_forest_classifier
    # this test will fail
    def test_random_forest_nan(self):
        res = self.c.random_forest_classifier(self.df16, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ######################### Rank Sum Tests ###############################
    def test_rank_sum_1(self):
        res = self.c.rank_sum(self.df7, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_rank_sum_nonnumeric(self):
        res = self.c.rank_sum(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    # TODO: handle Nan values in test_rank_sum
    # currently, rank sum checks if there *any* numeric columns. If there is at least one
    # numeric col, it still runs. Is this what we want? How do we want to handle nan vals?
    def test_rank_sum_nan(self):
        res = self.c.rank_sum(self.df16, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Scatterplot Regression Tests #######################
    # this plot looks nice! good sanity check
    def test_scatterplot_regression_1(self):
        res = self.c.scatterplot_regression(self.df_1, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_scatterplot_regression_nonnumeric(self):
        res = self.c.scatterplot_regression(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    def test_scatterplot_regression_nan(self):
        res = self.c.scatterplot_regression(self.df16, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Shuffle Split Tests #######################
    # NOTE: shuffle_split stores the resulting dataframe in res['output'], and res['result'] contains
    # a string. Is this what we want, or should 'result' be the df as well?
    def test_shuffle_split_1(self):
        res = self.c.shuffle_split(self.df_1, [], "", "")
        df = self.converter(res['output'])
        self.assertFalse(df.empty)

    def test_shuffle_split_nonnumeric(self):
        res = self.c.shuffle_split(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    def test_shuffle_split_nan(self):
        res = self.c.shuffle_split(self.df16, [], "", "")
        self.assertNotEqual(res['result'], "Dataframe has no numeric values")

    ###################### Stack FacetGrid Tests #######################
    # returns an img_lst like we want
    def test_stack_ftgrid_1(self):
        res = self.c.stack_ftgrid(self.df4, [], "", "")
        self.assertNotIsInstance(res['result'], Exception)

    def test_stack_ftgrid_onlynumeric(self):
        res = self.c.stack_ftgrid(self.df1, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric or category values")

    def test_stack_ftgrid_onlycategorical(self):
        res = self.c.stack_ftgrid(self.df5, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric or category values")

    ###################### Linear Regression Tests #######################
    # what df would work/ is expected behavior? df_1 doesn't work, but iris does
    # this test will error
    def test_lin_reg(self):
        res = self.c.test_linear_regression(self.df_1, [], "", "")
        df = self.converter(res['result'])

    def test_linear_regression_nonnumeric(self):
        res = self.c.test_linear_regression(self.df11, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

        res = self.c.test_linear_regression(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    # TODO: NaN handling for linear_regression. This test will fail
    # NOTE: We also need to do more input validation on the shape/size of the df so we can avoid errors like
    # "Cannot have number of splits greater than number of samples"
    def test_linear_regression_nan(self):
        res = self.c.test_linear_regression(self.df16, [], "", "")
        self.assertEqual(res['result'], "Dataframe has no numeric values")

    ###################### Top 5 Category Tests ###########################
    def test_top_five(self):
        # this returns an empty df, but i think it shouldn't
        res = self.c.top5cat(self.df7, [], "", "")
        df = self.converter(res['result'])

    def test_top_five_noncategorical(self):
        res = self.c.top5cat(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    ###################### Unique Column Tests ###########################
    def test_unique_cols(self):
        res = self.c.unique_column_values(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

        res = self.c.unique_column_values(self.df1, [], "", "")
        df = self.converter(res['result'])
        self.assertFalse(df.empty)
        self.assertEqual(df.size, 30)

        res = self.c.unique_column_values(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertLess(df.size, self.df7.size)

    ###################### Variance Tests ###########################
    def test_variance(self):
        res = self.c.variance(self.df7, [], "", "")
        df = self.converter(res['result'])
        self.assertEqual(df.shape[0], 4)

    def test_variance_nonnumeric(self):
        res = self.c.variance(self.df5, [], "", "")
        df = self.converter(res['result'])
        self.assertTrue(df.empty)

    ###################### Linear Regression Tests ###########################


if __name__ == '__main__':
    unittest.main()
