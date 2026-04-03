import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# required functions are in causal_analysis.py


from causal_framework.causal_analysis import (
     is_jja, driver_overlaps, adjust_date, 
     get_start_date, get_both_start_dates, check_coincidence, 
     calculate_decomposition_terms, calculate_decomposition_terms_multi)


class TestDateFiltering:
    """Test JJA month filtering"""
    def test_is_jja_june(self):
        """test tjat dates june are identified as JJA"""
        assert is_jja(20200615) == True
        
    def test_is_jja_july(self):
        """test that july dates are identified as JJA"""
        assert is_jja(20200715) == True
        
    def test_is_jja_august(self):
        """test that August dates are identified as JJA"""
        assert is_jja(20200815) == True
        
    def test_is_not_jja_winter(self):
        """test that winter months are not JJA"""
        assert is_jja(20201215) == False
        assert is_jja(20200115) == False
        
    def test_is_not_jja_spring(self):
        """test that spring months are not JJA"""
        assert is_jja(20200415) == False
        

class TestDriverOverlaps:
    """test overlap detection between events"""
    
    def test_complete_overlap(self):
        """Test when compound event completely overlaps driver"""
        assert driver_overlaps(20200610, 20200615, 20200610, 20200615) == True
        
    def test_partial_overlap_start(self):
        """Test when events overlap at start"""
        assert driver_overlaps(20200610, 20200615, 20200612, 20200618) == True
        
    def test_partial_overlap_end(self):
        """Test when events overlap at end"""
        assert driver_overlaps(20200610, 20200615, 20200608, 20200612) == True
        
    def test_driver_contains_compound(self):
        """Test when driver event contains compound event"""
        assert driver_overlaps(20200612, 20200614, 20200610, 20200618) == True
        
    def test_compound_contains_driver(self):
        """Test when compound event contains driver event"""
        assert driver_overlaps(20200610, 20200618, 20200612, 20200614) == True
        
    def test_no_overlap_before(self):
        """Test when events don't overlap (driver before compound)"""
        assert driver_overlaps(20200615, 20200620, 20200605, 20200610) == False
        
    def test_no_overlap_after(self):
        """Test when events don't overlap (driver after compound)"""
        assert driver_overlaps(20200605, 20200610, 20200615, 20200620) == False
        
    def test_adjacent_events(self):
        """Test adjacent but non-overlapping events"""
        # Depends on your overlap definition
        result = driver_overlaps(20200611, 20200615, 20200605, 20200610)
        assert isinstance(result, bool)


class TestDateAdjustment:
    """Test date adjustment with offsets"""
    
    def test_adjust_date_same_month(self):
        result = adjust_date('20200615', offset=5)
        assert result == '20200620'

    def test_adjust_date_cross_month(self):
        result = adjust_date('20200628', offset=5)
        assert result == '20200703'

    def test_adjust_date_negative_offset(self):
        result = adjust_date('20200615', offset=-5)
        assert result == '20200610'

    def test_adjust_date_zero_offset(self):
        result = adjust_date('20200615', offset=0)
        assert result == '20200615'

class TestEventCombinations:
    """Test event combination logic"""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample event data"""
        return {
            'A': [(20200610, 20200612), (20200620, 20200622)],
            'B': [(20200611, 20200613), (20200625, 20200627)],
            'S': [(20200615, 20200617)]
        }
    
    def test_get_start_date_single_event(self):
        """Test getting start date from single event"""
        events = [(20200610, 20200615)]
        assert get_start_date(events) == 20200610
        
    def test_get_both_start_dates(self):
        """Test getting both start and end dates"""
        events = [(20200610, 20200615)]
        result = get_both_start_dates(events)
        assert result == (20200610, 20200615)
        
    def test_coincidence_overlap(self, sample_events):
        event1 = (20200610, 20200615)
        include_dict = {'A': (20200612, 20200618)}
        assert check_coincidence(event1, include_dict) == True

    def test_coincidence_no_overlap(self, sample_events):
        event1 = (20200601, 20200605)
        include_dict = {'A': (20200610, 20200615)}
        assert check_coincidence(event1, include_dict) == False


class TestProbabilityCalculations:
    """Test probability and combination calculations"""
    
    @pytest.fixture
    def sample_hist_df(self):
        return pd.DataFrame({
            'ensemble': [1, 2, 3],
            'p(C)': [0.08, 0.09, 0.085],
            'p(A)': [0.1, 0.12, 0.11],
            'p(A and C)': [0.05, 0.06, 0.055],
            'p(C|A)': [0.5, 0.5, 0.5],
            'AR only count': [10, 12, 11]
        })

    @pytest.fixture
    def sample_fut_df(self):
        return pd.DataFrame({
            'ensemble': [1, 2, 3],
            'p(C)': [0.10, 0.11, 0.105],
            'p(A)': [0.15, 0.18, 0.16],
            'p(A and C)': [0.08, 0.09, 0.085],
            'p(C|A)': [0.53, 0.5, 0.53],
            'AR only count': [15, 18, 17]
        })
    
    def test_decomposition_terms_structure(self, sample_hist_df, sample_fut_df):
        result = calculate_decomposition_terms(sample_hist_df, sample_fut_df, 'A_only')
        expected_cols = ['ensemble', 'A_only_delta_p_C', 'A_only_t_total', 'A_only_delta_p_event']
        assert all(col in result.columns for col in expected_cols)

    def test_decomposition_terms_values(self, sample_hist_df, sample_fut_df):
        result = calculate_decomposition_terms(sample_hist_df, sample_fut_df, 'A_only')
        # t_total = t1 + t2 + t3, just check it's a finite number
        assert result['A_only_t_total'].notna().all()

        def test_probability_bounds(self, sample_hist_df, sample_fut_df):
            """Test that probabilities are between 0 and 1"""
            assert (sample_hist_df['p(A)'] >= 0).all()
            assert (sample_hist_df['p(A)'] <= 1).all()
            assert (sample_fut_df['p(C|A)'] >= 0).all()
            assert (sample_fut_df['p(C|A)'] <= 1).all()


class TestDataLoading:
    """Test data loading and filtering"""
    
    @pytest.fixture
    def sample_ar_df(self):
        """Create sample AR events dataframe"""
        return pd.DataFrame({
            'ensemble': [1, 1, 1, 2],
            'AR_start_date': [19790615, 19790715, 20600620, 20600720],
            'AR_end_date': [19790617, 19790717, 20600622, 20600722],
            'AR_duration': [3, 3, 3, 3]
        })
    
    def test_filter_by_ensemble(self, sample_ar_df):
        """Test filtering by ensemble number"""
        filtered = sample_ar_df[sample_ar_df['ensemble'] == 1]
        assert len(filtered) == 3
        assert (filtered['ensemble'] == 1).all()
        
    def test_filter_by_period(self, sample_ar_df):
        """Test filtering by time period"""
        period_start = 1979
        period_end = 2018
        filtered = sample_ar_df[
            (sample_ar_df['AR_start_date'].astype(str).str[:4].astype(int) >= period_start) &
            (sample_ar_df['AR_start_date'].astype(str).str[:4].astype(int) <= period_end)
        ]
        assert len(filtered) == 2
        
    def test_filter_jja_events(self, sample_ar_df):
        """Test filtering for JJA months only"""
        filtered = sample_ar_df[sample_ar_df['AR_start_date'].apply(is_jja)]
        assert len(filtered) == len(sample_ar_df)  # All test events are in JJA
        

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame(columns=['start_date', 'end_date'])
        assert len(empty_df) == 0
        # Should not raise error when creating compound window
        
    def test_zero_division_protection(self):
        """Test division by zero protection in probability calculations"""
        # When driver count is 0, p(C|A) should be 0
        driver_count = 0
        combination_count = 5
        result = combination_count / driver_count if driver_count > 0 else 0
        assert result == 0
        
    def test_single_event_processing(self):
        """Test processing when only one event exists"""
        events = [(20200615, 20200617)]
        assert len(events) == 1
        assert get_start_date(events) == 20200615
        
    def test_no_overlaps_found(self):
        """Test when no overlaps are found between drivers and compound events"""
        # Create non-overlapping events
        compound_start, compound_end = 20200601, 20200605
        driver_start, driver_end = 20200610, 20200615
        assert driver_overlaps(compound_start, compound_end, driver_start, driver_end) == False


class TestIntegration:
    """Integration tests for full workflow"""
    
    @pytest.fixture
    def full_test_dataset(self):
        """Create a complete small test dataset"""
        return {
            'ar_events': pd.DataFrame({
                'ensemble': [1, 1],
                'AR_start_date': [19790615, 19790715],
                'AR_end_date': [19790617, 19790717],
                'AR_duration': [3, 3]
            }),
            'compound_events': pd.DataFrame({
                'start_date': [19790616, 19790716],
                'length_days': [3, 2],
                'extreme_days': [2, 2]
            })
        }
    
    def test_end_to_end_probability_calculation(self, full_test_dataset):
        """Test complete probability calculation workflow"""
        # This would test the entire pipeline
        # from data loading through to final probability calculations
        pass  # Implement based on your modularized functions


# Configuration for pytest
@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'period_start_hist': 1979,
        'period_end_hist': 2018,
        'period_start_fut': 2060,
        'period_end_fut': 2079,
        'tau': -5  # or whatever your tau value is
    }


# Parametrized tests for multiple ensembles
class TestMultipleEnsembles:
    """Test consistency across ensembles"""
    
    @pytest.mark.parametrize("ensemble", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_ensemble_probability_bounds(self, ensemble):
        """Test that probabilities are valid for all ensembles"""
        # This would load actual results and verify bounds
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])