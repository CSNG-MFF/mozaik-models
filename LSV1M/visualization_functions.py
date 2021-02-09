import pylab
import numpy

import mozaik.storage.queries as queries
import matplotlib.gridspec as gridspec
from mozaik.visualization.plotting import Plotting
from mozaik.visualization.helper_functions import *
from parameters import ParameterSet
from mozaik.storage.queries import *
from mozaik.analysis.analysis import *
from mozaik.controller import Global
from mozaik.visualization.plotting import (Plotting, GSynPlot, RasterPlot, PerNeuronAnalogSignalScatterPlot,
                                           VmPlot, ConductanceSignalListPlot, ScatterPlot,
                                           AnalogSignalListPlot, OverviewPlot, PerNeuronValueScatterPlot, PlotTuningCurve, PerNeuronValuePlot, CorticalColumnRasterPlot)
from mozaik.visualization.simple_plot import *

from mozaik.tools.circ_stat import circular_dist
import mozaik.visualization.helper_functions as phf

import mozaik
logger = mozaik.getMozaikLogger()


class MRfigReal(Plotting):
    required_parameters = ParameterSet({
        'SimpleSheetName': str,  # the name of the sheet for which to plot
        'ComplexSheetName': str,  # which neuron to show
    })

    def plot(self):
        self.fig = pylab.figure(facecolor='w', **self.fig_param)
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.07, right=0.97, top=0.9, bottom=0.1)
        gs = gs[0, 0]

        dsv_l4 = self.datastore.get_analysis_result(identifier='PerNeuronValue', sheet_name=self.parameters.SimpleSheetName,
                                                    analysis_algorithm='ModulationRatio', value_name='Modulation ratio(time)')
        dsv_l23 = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', sheet_name=self.parameters.ComplexSheetName, analysis_algorithm='ModulationRatio', value_name='Modulation ratio(time)')

        dsv = queries.param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=0)
        dsv_l4_v_F0 = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name=self.parameters.SimpleSheetName, value_name='-(x+y)(F0_Vm,Mean(VM))')
        dsv_l23_v_F0 = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name=self.parameters.ComplexSheetName, value_name='-(x+y)(F0_Vm,Mean(VM))')
        dsv_l4_v_F1 = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name=self.parameters.SimpleSheetName, value_name='F1_Vm')
        dsv_l23_v_F1 = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name=self.parameters.ComplexSheetName, value_name='F1_Vm')

        dsv_l4_v_F0_inh = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name='V1_Inh_L4', value_name='-(x+y)(F0_Vm,Mean(VM))')
        dsv_l23_v_F0_inh = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name='V1_Inh_L2/3', value_name='-(x+y)(F0_Vm,Mean(VM))')
        dsv_l4_v_F1_inh = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name='V1_Inh_L4', value_name='F1_Vm')
        dsv_l23_v_F1_inh = dsv.get_analysis_result(
            identifier='PerNeuronValue', sheet_name='V1_Inh_L2/3', value_name='F1_Vm')

        assert len(dsv_l4) == 1,  str(len(dsv_l4))
        assert len(dsv_l4_v_F0) == 1
        assert len(dsv_l4_v_F1) == 1
        if self.parameters.ComplexSheetName != 'None':
            assert len(dsv_l23) == 1
            assert len(dsv_l23_v_F0) == 1
            assert len(dsv_l23_v_F1) == 1

        l4_ids = dsv_l4_v_F0[0].ids
        if self.parameters.ComplexSheetName != 'None':
            l23_ids = dsv_l23_v_F0[0].ids

        l4_ids_inh = dsv_l4_v_F0_inh[0].ids
        l23_ids_inh = dsv_l23_v_F0_inh[0].ids

        l4_exc_or = self.datastore.full_datastore.get_analysis_result(identifier='PerNeuronValue', value_name=[
                                                                      'LGNAfferentOrientation', 'ORMapOrientation'], sheet_name='V1_Exc_L4')[0]
        l4_ids = numpy.array(l4_ids)[numpy.nonzero(numpy.array([circular_dist(
            l4_exc_or.get_value_by_id(i), 0, numpy.pi) for i in l4_ids]) < 0.4)[0]]

        if self.parameters.ComplexSheetName != 'None':
            l23_exc_or = self.datastore.full_datastore.get_analysis_result(identifier='PerNeuronValue', value_name=[
                                                                           'LGNAfferentOrientation', 'ORMapOrientation'], sheet_name='V1_Exc_L2/3')[0]
            l23_ids = numpy.array(l23_ids)[numpy.nonzero(numpy.array([circular_dist(
                l23_exc_or.get_value_by_id(i), 0, numpy.pi) for i in l23_ids]) < 0.4)[0]]

        l4_exc_or_inh = self.datastore.full_datastore.get_analysis_result(identifier='PerNeuronValue', value_name=[
                                                                          'LGNAfferentOrientation', 'ORMapOrientation'], sheet_name='V1_Inh_L4')[0]
        l4_ids_inh = numpy.array(l4_ids_inh)[numpy.nonzero(numpy.array([circular_dist(
            l4_exc_or_inh.get_value_by_id(i), 0, numpy.pi) for i in l4_ids_inh]) < 0.4)[0]]

        l23_exc_or_inh = self.datastore.full_datastore.get_analysis_result(identifier='PerNeuronValue', value_name=[
                                                                           'LGNAfferentOrientation', 'ORMapOrientation'], sheet_name='V1_Inh_L2/3')[0]
        l23_ids_inh = numpy.array(l23_ids_inh)[numpy.nonzero(numpy.array([circular_dist(
            l23_exc_or_inh.get_value_by_id(i), 0, numpy.pi) for i in l23_ids_inh]) < 0.4)[0]]

        l4_v_mr = numpy.array(dsv_l4_v_F1[0].get_value_by_id(
            l4_ids))/numpy.array(dsv_l4_v_F0[0].get_value_by_id(l4_ids))
        l4_v_mr_inh = numpy.array(dsv_l4_v_F1_inh[0].get_value_by_id(
            l4_ids_inh))/numpy.array(dsv_l4_v_F0_inh[0].get_value_by_id(l4_ids_inh))
        dsv_l4 = dsv_l4[0]
        if self.parameters.ComplexSheetName != 'None':
            l23_v_mr = numpy.array(dsv_l23_v_F1[0].get_value_by_id(
                l23_ids))/numpy.array(dsv_l23_v_F0[0].get_value_by_id(l23_ids))
            l23_v_mr_inh = numpy.array(dsv_l23_v_F1_inh[0].get_value_by_id(
                l23_ids_inh))/numpy.array(dsv_l23_v_F0_inh[0].get_value_by_id(l23_ids_inh))
            dsv_l23 = dsv_l23[0]

        dsv_simple = numpy.append(
            dsv_l4.values[dsv_l4.values < 1.0], dsv_l23.values[dsv_l23.values < 1.0])
        dsv_complex = numpy.append(
            dsv_l4.values[dsv_l4.values > 1.0], dsv_l23.values[dsv_l23.values > 1.0])

        simple_mr = numpy.append(numpy.array(dsv_l4.get_value_by_id(l4_ids))[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) < 1.0], numpy.array(dsv_l23.get_value_by_id(l23_ids))[numpy.array(dsv_l23.get_value_by_id(l23_ids)) < 1.0])
        complex_mr = numpy.append(numpy.array(dsv_l4.get_value_by_id(l4_ids))[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) > 1.0], numpy.array(dsv_l23.get_value_by_id(l23_ids))[numpy.array(dsv_l23.get_value_by_id(l23_ids)) > 1.0])

        simple_v_mr = numpy.append(l4_v_mr[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) < 1.0], l23_v_mr[numpy.array(dsv_l23.get_value_by_id(l23_ids)) < 1.0])
        complex_v_mr = numpy.append(l4_v_mr[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) > 1.0], l23_v_mr[numpy.array(dsv_l23.get_value_by_id(l23_ids)) > 1.0])

        dsv_simple_v_F0 = numpy.append(numpy.array(dsv_l4_v_F0[0].get_value_by_id(l4_ids))[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) < 1.0], numpy.array(dsv_l23_v_F0[0].get_value_by_id(l23_ids))[numpy.array(dsv_l23.get_value_by_id(l23_ids)) < 1.0])
        dsv_complex_v_F0 = numpy.append(numpy.array(dsv_l4_v_F0[0].get_value_by_id(l4_ids))[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) > 1.0], numpy.array(dsv_l23_v_F0[0].get_value_by_id(l23_ids))[numpy.array(dsv_l23.get_value_by_id(l23_ids)) > 1.0])

        dsv_simple_v_F1 = numpy.append(numpy.array(dsv_l4_v_F1[0].get_value_by_id(l4_ids))[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) < 1.0], numpy.array(dsv_l23_v_F1[0].get_value_by_id(l23_ids))[numpy.array(dsv_l23.get_value_by_id(l23_ids)) < 1.0])
        dsv_complex_v_F1 = numpy.append(numpy.array(dsv_l4_v_F1[0].get_value_by_id(l4_ids))[numpy.array(dsv_l4.get_value_by_id(
            l4_ids)) > 1.0], numpy.array(dsv_l23_v_F1[0].get_value_by_id(l23_ids))[numpy.array(dsv_l23.get_value_by_id(l23_ids)) > 1.0])

        gs = gridspec.GridSpecFromSubplotSpec(
            3, 7, subplot_spec=gs, wspace=0.3)
        ax = pylab.subplot(gs[0, 0])
        ax.hist(dsv_l4.values, bins=numpy.arange(
            0, 2.01, 0.2), color='gray', rwidth=0.8)
        disable_top_right_axis(ax)
        disable_left_axis(ax)
        pylab.ylim(0, 540)
        disable_xticks(ax)
        remove_x_tick_labels()
        remove_y_tick_labels()
        pylab.ylabel('Layer 4', fontsize=19)
        ax = pylab.subplot(gs[1, 0])
        if self.parameters.ComplexSheetName != 'None':
            ax.hist(dsv_l23.values, bins=numpy.arange(
                0, 2.01, 0.2), color='gray', rwidth=0.8)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            pylab.ylim(0, 540)
            disable_xticks(ax)
            remove_x_tick_labels()
            remove_y_tick_labels()
            pylab.ylabel('Layer 2/3', fontsize=19)

            ax = pylab.subplot(gs[2, 0])
            ax.hist([dsv_complex, dsv_simple], bins=numpy.arange(
                0, 2.01, 0.2), histtype='barstacked', color=['w', 'k'], rwidth=0.8, ec='black')
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            pylab.ylim(0, 540)
            pylab.ylabel('Pooled', fontsize=19)
            three_tick_axis(ax.xaxis)
            remove_y_tick_labels()
            pylab.xlabel('F1/F0 spikes', fontsize=19)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(19)
            disable_top_right_axis(ax)
            disable_left_axis(ax)

        ax = pylab.subplot(gs[0, 1])
        ax.hist(l4_v_mr, bins=numpy.arange(
            0, 3.01, 0.3), color='gray', rwidth=0.8)
        disable_top_right_axis(ax)
        disable_left_axis(ax)
        disable_xticks(ax)
        remove_x_tick_labels()
        remove_y_tick_labels()
        pylab.xlim(0, 3.0)

        if self.parameters.ComplexSheetName != 'None':
            ax = pylab.subplot(gs[1, 1])
            ax.hist(l23_v_mr, bins=numpy.arange(
                0, 3.01, 0.3), color='gray', rwidth=0.8)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            disable_xticks(ax)
            remove_x_tick_labels()
            remove_y_tick_labels()
            pylab.xlim(0, 3.0)
            ax = pylab.subplot(gs[2, 1])
            ax.hist([complex_v_mr, simple_v_mr], bins=numpy.arange(
                0, 3.01, 0.3), histtype='barstacked', color=['w', 'k'], rwidth=0.8, ec='black')
            three_tick_axis(ax.xaxis)
            remove_y_tick_labels()
            pylab.xlabel('F1/F0 Vm', fontsize=19)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(19)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            pylab.xlim(0, 3.0)

        ax = pylab.subplot(gs[0, 2])
        ax.hist(numpy.abs(dsv_l4_v_F0[0].get_value_by_id(l4_ids)), bins=numpy.arange(
            0, 4.01, 0.4), color='gray', rwidth=0.8)
        disable_top_right_axis(ax)
        disable_left_axis(ax)
        disable_left_axis(ax)
        disable_xticks(ax)
        remove_x_tick_labels()
        remove_y_tick_labels()
        pylab.xlim(0, 4.0)

        if self.parameters.ComplexSheetName != 'None':
            ax = pylab.subplot(gs[1, 2])
            ax.hist(numpy.abs(dsv_l23_v_F0[0].get_value_by_id(l23_ids)), bins=numpy.arange(
                0, 4.01, 0.4), color='gray', rwidth=0.8)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            disable_xticks(ax)
            remove_x_tick_labels()
            remove_y_tick_labels()
            pylab.xlim(0, 4.0)
            ax = pylab.subplot(gs[2, 2])
            ax.hist([numpy.abs(dsv_complex_v_F0), numpy.abs(dsv_simple_v_F0)], bins=numpy.arange(
                0, 4.01, 0.4), histtype='barstacked', color=['w', 'k'], rwidth=0.8, ec='black')
            three_tick_axis(ax.xaxis)
            remove_y_tick_labels()
            pylab.xlabel('F0 Vm (mV)', fontsize=19)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(19)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            pylab.xlim(0, 4.0)

        ax = pylab.subplot(gs[0, 3])
        ax.hist(numpy.abs(dsv_l4_v_F1[0].get_value_by_id(l4_ids)), bins=numpy.arange(
            0, 5.01, 0.5), color='gray', rwidth=0.8)
        disable_top_right_axis(ax)
        disable_left_axis(ax)
        disable_xticks(ax)
        remove_x_tick_labels()
        remove_y_tick_labels()
        pylab.xlim(0, 5.0)

        if self.parameters.ComplexSheetName != 'None':
            ax = pylab.subplot(gs[1, 3])
            ax.hist(numpy.abs(dsv_l23_v_F1[0].get_value_by_id(l23_ids)), bins=numpy.arange(
                0, 5.01, 0.5), color='gray', rwidth=0.8)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            disable_xticks(ax)
            remove_x_tick_labels()
            remove_y_tick_labels()
            pylab.xlim(0, 5.0)
            ax = pylab.subplot(gs[2, 3])
            ax.hist([numpy.abs(dsv_complex_v_F1), numpy.abs(dsv_simple_v_F1)], bins=numpy.arange(
                0, 5.01, 0.5), histtype='barstacked', color=['w', 'k'], rwidth=0.8, ec='black')
            three_tick_axis(ax.xaxis)
            remove_y_tick_labels()
            pylab.xlabel('F1 Vm (mV)', fontsize=19)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(19)
            disable_top_right_axis(ax)
            disable_left_axis(ax)
            pylab.xlim(0, 5.0)

        logger.info(len(simple_v_mr))
        logger.info(len(dsv_simple))
        if self.parameters.ComplexSheetName != 'None':
            ggs = gridspec.GridSpecFromSubplotSpec(20, 20, gs[:, 4:7])
            ax = pylab.subplot(ggs[3:18, 3:18])
            ax.plot(complex_v_mr, complex_mr, 'ok', label='layer 2/3')
            ax.plot(simple_v_mr, simple_mr, 'ok', label='layer 4')
            pylab.xlabel('F1/F0 Vm', fontsize=19)
            pylab.ylabel('F1/F0 Spikes', fontsize=19)
            pylab.xlim(0, 3.0)
            pylab.ylim(0, 2.0)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(19)

        if self.plot_file_name:
            pylab.savefig(Global.root_directory+self.plot_file_name)


class LSV1MReponseOverview(Plotting):
    required_parameters = ParameterSet({
        'l4_exc_neuron': int,
        'l4_inh_neuron': int,
        'l23_exc_neuron': int,
        'l23_inh_neuron': int,
    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            19, 68, subplot_spec=subplotspec, hspace=1.0, wspace=100.0)

        dsv = param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[0], st_contrast=[100])
        plots['ExcOr0L4'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': self.parameters.l4_exc_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[0:9, 0:16], {'x_label': None, 'x_axis': False, 'x_ticks': False})

        dsv = param_filter_query(self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[
                                 numpy.pi/2], st_contrast=[100])
        plots['ExcOrPiHL4'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': self.parameters.l4_exc_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[0:9, 17:33], {'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False, 'y_axis': False, 'y_ticks': False})

        dsv = param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[0], st_contrast=[100])
        plots['InhOr0L4'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': self.parameters.l4_inh_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[0:9, 35:51], {'x_label': None, 'y_label': None, 'title': None, 'y_axis': False, 'y_ticks': False, 'x_axis': False, 'x_ticks': False})

        dsv = param_filter_query(self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[
                                 numpy.pi/2], st_contrast=[100])
        plots['InhOrPiHL4'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': self.parameters.l4_inh_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[0:9, 52:68], {'x_label': None, 'y_label': None, 'title': None, 'y_axis': False, 'y_ticks': False, 'x_axis': False, 'x_ticks': False})

        def stim_plot(phase):
            x = numpy.arange(0, 2, 0.01)
            pylab.plot(x, numpy.sin(x*numpy.pi*4+phase), 'g', linewidth=3)
            phf.disable_top_right_axis(self.axis)
            phf.disable_xticks(self.axis)
            phf.disable_yticks(self.axis)
            phf.remove_x_tick_labels()
            phf.remove_y_tick_labels()
            self.axis.set_ylim(-1.3, 1.3)

        self.axis = pylab.subplot(gs[9:10, 0:16])
        stim_plot(0)
        self.axis = pylab.subplot(gs[9:10, 35:51])
        stim_plot(3.9)

        dsv = param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[0], st_contrast=[100])
        plots['ExcOr0L23'] = (OverviewPlot(dsv, ParameterSet(
            {'sheet_name': 'V1_Exc_L2/3', 'neuron': self.parameters.l23_exc_neuron, 'sheet_activity': {}, 'spontaneous': False})), gs[10:, 0:16], {'title': None})

        dsv = param_filter_query(self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[
                                 numpy.pi/2], st_contrast=[100])
        plots['ExcOrPiHL23'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': self.parameters.l23_exc_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[10:, 17:33], {'y_label': None, 'title': None, 'y_axis': False, 'y_ticks': False})

        dsv = param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[0], st_contrast=[100])
        plots['InhOr0L23'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': self.parameters.l23_inh_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[10:, 35:51], {'y_label': None, 'title': None, 'y_axis': False, 'y_ticks': False})

        dsv = param_filter_query(self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=[
                                 numpy.pi/2], st_contrast=[100])
        plots['InhOrPiHL23'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': self.parameters.l23_inh_neuron, 'sheet_activity': {
        }, 'spontaneous': False})), gs[10:, 52:68], {'y_label': None, 'title': None, 'y_axis': False, 'y_ticks': False})

        return plots


class SpontActOverview(Plotting):
    required_parameters = ParameterSet({
        'l4_exc_neuron': int,
        'l4_inh_neuron': int,
        'l23_exc_neuron': int,
        'l23_inh_neuron': int,
    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            8, 3, subplot_spec=subplotspec, hspace=0.3, wspace=0.45)
        dsv = param_filter_query(
            self.datastore, st_direct_stimulation_name=None, st_name=['InternalStimulus'])

        fontsize = 17

        analog_ids1 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name='V1_Exc_L4').get_segments()[0].get_stored_esyn_ids()))

        tstop = queries.param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name="InternalStimulus",
                                           sheet_name='V1_Exc_L4').get_segments()[0].get_vm(analog_ids1[0]).t_stop.magnitude
        tstop = min(min(tstop, 5.0), tstop)

        spike_ids = param_filter_query(self.datastore, sheet_name="V1_Exc_L4").get_segments()[
            0].get_stored_spike_train_ids()
        spike_ids_inh = param_filter_query(self.datastore, sheet_name="V1_Inh_L4").get_segments()[
            0].get_stored_spike_train_ids()
        if self.parameters.l23_exc_neuron != -1:
            spike_ids23 = param_filter_query(
                self.datastore, sheet_name="V1_Exc_L2/3").get_segments()[0].get_stored_spike_train_ids()
            spike_ids_inh23 = param_filter_query(
                self.datastore, sheet_name="V1_Inh_L2/3").get_segments()[0].get_stored_spike_train_ids()

        if self.parameters.l23_exc_neuron != -1:
            d = int(numpy.min([numpy.floor(len(spike_ids)/4.0), len(spike_ids_inh),
                               numpy.floor(len(spike_ids23)/4.0), len(spike_ids_inh23)]))
            neuron_ids = [spike_ids_inh[:d], spike_ids[:d*4],
                          spike_ids_inh23[:d], spike_ids23[:d*4]]
        else:
            d = int(
                numpy.min([numpy.floor(len(spike_ids)/4.0), len(spike_ids_inh)]))
            neuron_ids = [spike_ids_inh[:d], spike_ids[:d*4]]

        if self.parameters.l23_exc_neuron != -1:
            plots['SpikingOverview'] = (CorticalColumnRasterPlot(dsv, ParameterSet({'spontaneous': False, 'sheet_names': ['V1_Inh_L4', 'V1_Exc_L4', 'V1_Inh_L2/3', 'V1_Exc_L2/3'], 'neurons': neuron_ids, 'colors': [
                                        '#0000FF', '#FF0000', '#0000FF', '#FF0000'], 'labels': ["L4i", "L4e", "L2/3i", "L2/3e"]})), gs[:, 0], {'fontsize': fontsize, 'x_lim': (0, tstop)})
            plots['ExcL2/3Cond'] = (GSynPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': self.parameters.l23_exc_neuron, 'spontaneous': False})), gs[0, 1:], {
                                    'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_lim': (0, 25), 'y_lim': (0, 25), 'y_axis': None})
            plots['ExcL2/3Vm'] = (VmPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L2/3', 'neuron': self.parameters.l23_exc_neuron, 'spontaneous': False})), gs[1, 1:], {
                                  'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_axis': None})
            plots['InhL2/3Cond'] = (GSynPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': self.parameters.l23_inh_neuron, 'spontaneous': False})), gs[2, 1:], {
                                    'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_lim': (0, 25), 'y_axis': None})
            plots['InhL2/3Vm'] = (VmPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L2/3', 'neuron': self.parameters.l23_inh_neuron, 'spontaneous': False})), gs[3, 1:], {
                                  'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_axis': None})
        else:
            plots['SpikingOverview'] = (CorticalColumnRasterPlot(dsv, ParameterSet({'spontaneous': False, 'sheet_names': ['V1_Inh_L4', 'V1_Exc_L4'], 'neurons': neuron_ids, 'colors': [
                                        '#666666', '#000000'], 'labels': ["L4i", "L4e"]})), gs[:, 0], {'fontsize': fontsize, 'x_lim': (0, tstop)})

        plots['ExcL4Cond'] = (GSynPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': self.parameters.l4_exc_neuron, 'spontaneous': False})), gs[4, 1:], {
                              'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_lim': (0, 25), 'y_axis': None})
        plots['ExcL4Vm'] = (VmPlot(dsv, ParameterSet({'sheet_name': 'V1_Exc_L4', 'neuron': self.parameters.l4_exc_neuron, 'spontaneous': False})), gs[5, 1:], {
                            'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_axis': None})
        plots['InhL4Cond'] = (GSynPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': self.parameters.l4_inh_neuron, 'spontaneous': False})), gs[6, 1:], {
                              'x_label': None, 'fontsize': fontsize, 'x_ticks': [], 'title': None, 'x_lim': (0, tstop), 'y_lim': (0, 25)})
        plots['InhL4Vm'] = (VmPlot(dsv, ParameterSet({'sheet_name': 'V1_Inh_L4', 'neuron': self.parameters.l4_inh_neuron, 'spontaneous': False})), gs[7, 1:], {
                            'fontsize': fontsize, 'title': None, 'x_ticks': None, 'x_lim': (0, tstop)})

        return plots


class SpontStatisticsOverview(Plotting):
    required_parameters = ParameterSet({

    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            12, 4, subplot_spec=subplotspec, hspace=10.0, wspace=0.5)
        dsv = param_filter_query(
            self.datastore, st_direct_stimulation_name=None, st_name=['InternalStimulus'])

        l23_flag = len(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                          sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Mean(Firing rate)').get_analysis_result()) != 0

        fontsize = 17

        mean_firing_rate_L4E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                  sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
        mean_firing_rate_L4I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                  sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
        std_firing_rate_L4E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                            sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)
        std_firing_rate_L4I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                            sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)

        if l23_flag:
            mean_firing_rate_L23E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                       sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
            mean_firing_rate_L23I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                       sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
            std_firing_rate_L23E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                                 sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)
            std_firing_rate_L23I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                                 sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)
        else:
            mean_firing_rate_L23E = 0
            mean_firing_rate_L23I = 0
            std_firing_rate_L23E = 0
            std_firing_rate_L23I = 0

        logger.info('mean_firing_rate_L4E :' + str(mean_firing_rate_L4E))
        logger.info('mean_firing_rate_L4I :' + str(mean_firing_rate_L4I))
        logger.info('mean_firing_rate_L23E :' + str(mean_firing_rate_L23E))
        logger.info('mean_firing_rate_L23I :' + str(mean_firing_rate_L23I))

        mean_CV_L4E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                         sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Mean(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value
        mean_CV_L4I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                         sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Mean(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value
        std_CV_L4E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                   sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Var(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value)
        std_CV_L4I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                   sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Var(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value)

        if l23_flag:
            mean_CV_L23E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                              sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Mean(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value
            mean_CV_L23I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                              sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Mean(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value
            std_CV_L23E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                        sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Var(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value)
            std_CV_L23I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                        sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Var(CV of ISI squared)', ads_unique=True).get_analysis_result()[0].value)
        else:
            mean_CV_L23E = 0
            mean_CV_L23I = 0
            std_CV_L23E = 0
            std_CV_L23I = 0

        logger.info('mean_CV_L4E :' + str(mean_CV_L4E))
        logger.info('mean_CV_L4I :' + str(mean_CV_L4I))
        logger.info('mean_CV_L23E :' + str(mean_CV_L23E))
        logger.info('mean_CV_L23I :' + str(mean_CV_L23I))

        mean_CC_L4E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                         sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Mean(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value
        mean_CC_L4I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                         sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Mean(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value
        std_CC_L4E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                   sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Var(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value)
        std_CC_L4I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                   sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Var(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value)
        if l23_flag:
            mean_CC_L23E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                              sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Mean(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value
            mean_CC_L23I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                              sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Mean(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value
            std_CC_L23E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                        sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Var(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value)
            std_CC_L23I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                        sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Var(Correlation coefficient(psth (bin=10.0)))', ads_unique=True).get_analysis_result()[0].value)
        else:
            mean_CC_L23E = 0
            mean_CC_L23I = 0
            std_CC_L23E = 0
            std_CC_L23I = 0

        logger.info('mean_CC_L4E :' + str(mean_CC_L4E))
        logger.info('mean_CC_L4I :' + str(mean_CC_L4I))
        logger.info('mean_CC_L23E :' + str(mean_CC_L23E))
        logger.info('mean_CC_L23I :' + str(mean_CC_L23I))

        def ms(a): return (numpy.mean(a), numpy.std(a))
        mean_VM_L4E, std_VM_L4E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L4', st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
        mean_VM_L4I, std_VM_L4I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L4', st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
        if l23_flag:
            mean_VM_L23E, std_VM_L23E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L2/3', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
            mean_VM_L23I, std_VM_L23I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L2/3', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
        else:
            mean_VM_L23E, std_VM_L23E = 0, 0
            mean_VM_L23I, std_VM_L23I = 0, 0
        logger.info('mean_VM_L4E :' + str(mean_VM_L4E))
        logger.info('mean_VM_L4I :' + str(mean_VM_L4I))
        logger.info('mean_VM_L23E :' + str(mean_VM_L23E))
        logger.info('mean_VM_L23I :' + str(mean_VM_L23I))

        mean_CondE_L4E, std_CondE_L4E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
        mean_CondE_L4I, std_CondE_L4I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
        if l23_flag:
            mean_CondE_L23E, std_CondE_L23E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
            mean_CondE_L23I, std_CondE_L23I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
        else:
            mean_CondE_L23E, std_CondE_L23E = 0, 0
            mean_CondE_L23I, std_CondE_L23I = 0, 0

        logger.info('mean_ECond :' + str(mean_CondE_L4E +
                                         mean_CondE_L4I+mean_CondE_L23E+mean_CondE_L23I))

        mean_CondI_L4E, std_CondI_L4E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
        mean_CondI_L4I, std_CondI_L4I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
        if l23_flag:
            mean_CondI_L23E, std_CondI_L23E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
            mean_CondI_L23I, std_CondI_L23I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
        else:
            mean_CondI_L23E, std_CondI_L23E = 0, 0
            mean_CondI_L23I, std_CondI_L23I = 0, 0

        logger.info('mean_ICond :' + str(mean_CondI_L4E +
                                         mean_CondI_L4I+mean_CondI_L23E+mean_CondI_L23I))

        pylab.rc('axes', linewidth=1)

        def plot_with_log_normal_fit(values, gs1, gs2, x_label=False, y_label=""):
            valuesnz = values[numpy.nonzero(values)[0]]
            h, bin_edges = numpy.histogram(numpy.log10(
                valuesnz), range=(-2, 2), bins=20, normed=True)
            bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.0

            m = numpy.mean(numpy.log10(valuesnz))
            nm = numpy.mean(valuesnz)
            s = numpy.std(numpy.log10(valuesnz))
            if s == 0:
                s = 1.0

            pylab.subplot(gs1)
            pylab.plot(numpy.logspace(-2, 2, 100), numpy.exp(-((numpy.log10(numpy.logspace(-2, 2, 100))-m)
                                                               ** 2)/(2*s*s))/(s*numpy.sqrt(2*numpy.pi)), linewidth=4, color="#666666")
            pylab.plot(numpy.power(10, bin_centers), h, 'ko', mec=None, mew=3)
            pylab.xlim(10**-2, 10**2)
            pylab.gca().set_xscale("log")
            if x_label:
                pylab.xlabel('firing rate [Hz]', fontsize=fontsize)
                pylab.xticks([0.01, 0.1, 1.0, 10, 100])
            else:
                pylab.xticks([])
            pylab.ylabel(y_label, fontsize=fontsize)
            pylab.yticks([0.0, 0.5, 1.0])
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())

            pylab.subplot(gs2)
            pylab.plot(numpy.logspace(-1, 2, 100), numpy.exp(-((numpy.log10(numpy.logspace(-1, 2, 100))-m)
                                                               ** 2)/(2*s*s))/(s*numpy.sqrt(2*numpy.pi)), linewidth=4, color="#666666")
            pylab.plot(numpy.logspace(-1, 2, 100),
                       numpy.exp(-numpy.logspace(-1, 2, 100)/nm)/nm, 'k--', linewidth=4)
            pylab.plot(numpy.power(10, bin_centers), h, 'ko', mec=None, mew=3)
            pylab.xlim(10**-1, 10**2)
            pylab.ylim(0.00001, 5.0)
            pylab.gca().set_xscale("log")
            pylab.gca().set_yscale("log")
            if x_label:
                pylab.xlabel('firing rate [Hz]', fontsize=fontsize)
                pylab.xticks([0.1, 1.0, 10, 100])
            else:
                pylab.xticks([])
            pylab.yticks([0.0001, 0.01, 1.0])
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())

        plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Exc_L4"], st_direct_stimulation_name=None, st_name=[
                                 'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[0:3, 2], gs[0:3, 3], y_label='L4e')
        plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Inh_L4"], st_direct_stimulation_name=None, st_name=[
                                 'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[3:6, 2], gs[3:6, 3], y_label='L4i')
        if l23_flag:
            plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Exc_L2/3"], st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[6:9, 2], gs[6:9, 3], y_label='L2/3e')
            plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Inh_L2/3"], st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[9:12, 2], gs[9:12, 3], x_label=True, y_label='L2/3i')

        def autolabel(rects, offset=0.25):
            # attach some text labels
            for rect in rects:
                height = rect.get_width()
                pylab.gca().text(rect.get_x() + rect.get_width() + abs(pylab.gca().get_xlim()[0] - pylab.gca().get_xlim()[1])*offset, rect.get_y()+0.012,
                                 '%.2g' % float(height),
                                 ha='center', va='bottom', fontsize=17)

        if True:
            pylab.subplot(gs[0:4, 0])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_firing_rate_L4E, mean_firing_rate_L23E], height=0.12, color='#000000', xerr=[
                            std_firing_rate_L4E, std_firing_rate_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_firing_rate_L4I, mean_firing_rate_L23I], height=0.12, color='#FFFFFF', xerr=[
                            std_firing_rate_L4I, std_firing_rate_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 8.0)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('firing rate (Hz)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.subplot(gs[4:8, 0])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CV_L4E, mean_CV_L23E], height=0.12, color='#000000', xerr=[
                            std_CV_L4E, std_CV_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CV_L4I, mean_CV_L23I], height=0.12, color='#FFFFFF', xerr=[
                            std_CV_L4I, std_CV_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 2.0)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('irregularity', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1, offset=0.37)
            autolabel(r2, offset=0.37)

            pylab.subplot(gs[8:12, 0])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CC_L4E, mean_CC_L23E], height=0.12, color='#000000', xerr=[
                            std_CC_L4E, std_CC_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CC_L4I, mean_CC_L23I], height=0.12, color='#FFFFFF', xerr=[
                            std_CC_L4I, std_CC_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 0.3)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('synchrony', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1, offset=0.6)
            autolabel(r2, offset=0.6)

            pylab.subplot(gs[0:4, 1])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [abs(mean_VM_L4E), numpy.abs(mean_VM_L23E)], height=0.12, color='#000000', xerr=[
                            std_VM_L4E, std_VM_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [abs(mean_VM_L4I), numpy.abs(mean_VM_L23I)], height=0.12, color='#FFFFFF', xerr=[
                            std_VM_L4I, std_VM_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            pylab.ylim(0, 1.0)
            pylab.xlim(40, 80)
            pylab.xticks([40, 60, 80], [-40, -60, -80])
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('membrane potential (mV)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.subplot(gs[4:8, 1])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CondE_L4E*1000, mean_CondE_L23E*1000], height=0.12, color='#000000', xerr=[
                            std_CondE_L4E*1000, std_CondE_L23E*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CondE_L4I*1000, mean_CondE_L23I*1000], height=0.12, color='#FFFFFF', xerr=[
                            std_CondE_L4I*1000, std_CondE_L23I*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 2.0)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('excitatory conductance (nS)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.subplot(gs[8:12, 1])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CondI_L4E*1000, mean_CondI_L23E*1000], height=0.12, color='#000000', xerr=[
                            std_CondI_L4E*1000, std_CondI_L23E*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CondI_L4I*1000, mean_CondI_L23I*1000], height=0.12, color='#FFFFFF', xerr=[
                            std_CondI_L4I*1000, std_CondI_L23I*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2), ec='k')
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 10)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('inhibitory conductance (nS)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.rc('axes', linewidth=1)

        return plots


class SpontStatisticsOverviewNew(Plotting):
    required_parameters = ParameterSet({

    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            12, 4, subplot_spec=subplotspec, hspace=10.0, wspace=0.5)
        dsv = param_filter_query(
            self.datastore, st_direct_stimulation_name=None, st_name=['InternalStimulus'])

        l23_flag = len(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                          sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Mean(Firing rate)').get_analysis_result()) != 0

        fontsize = 17

        mean_firing_rate_L4E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                  sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
        mean_firing_rate_L4I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                  sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
        std_firing_rate_L4E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                            sheet_name='V1_Exc_L4', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)
        std_firing_rate_L4I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                            sheet_name='V1_Inh_L4', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)

        if l23_flag:
            mean_firing_rate_L23E = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                       sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
            mean_firing_rate_L23I = param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                       sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Mean(Firing rate)', ads_unique=True).get_analysis_result()[0].value
            std_firing_rate_L23E = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                                 sheet_name='V1_Exc_L2/3', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)
            std_firing_rate_L23I = numpy.sqrt(param_filter_query(self.datastore, st_direct_stimulation_name=None, st_name='InternalStimulus', analysis_algorithm='PopulationMeanAndVar',
                                                                 sheet_name='V1_Inh_L2/3', identifier='SingleValue', value_name='Var(Firing rate)', ads_unique=True).get_analysis_result()[0].value)
        else:
            mean_firing_rate_L23E = 0
            mean_firing_rate_L23I = 0
            std_firing_rate_L23E = 0
            std_firing_rate_L23I = 0

        logger.info('mean_firing_rate_L4E :' + str(mean_firing_rate_L4E))
        logger.info('mean_firing_rate_L4I :' + str(mean_firing_rate_L4I))
        logger.info('mean_firing_rate_L23E :' + str(mean_firing_rate_L23E))
        logger.info('mean_firing_rate_L23I :' + str(mean_firing_rate_L23I))

        def mean_and_std(x): return (numpy.mean(x), numpy.std(x))

        s = queries.param_filter_query(
            self.datastore, st_name='InternalStimulus', sheet_name='V1_Exc_L4').get_segments()[0]
        isis = [numpy.diff(st.magnitude) for st in s.spiketrains]
        idxs = numpy.array([len(isi) for isi in isis]) > 5
        mean_CV_L4E, std_CV_L4E = mean_and_std(numpy.array(
            [numpy.std(isi)/numpy.mean(isi) for isi in isis])[idxs])
        s = queries.param_filter_query(self.datastore, st_name='InternalStimulus', sheet_name='V1_Exc_L4',
                                       value_name='Correlation coefficient(psth (bin=10.0))', ads_unique=True).get_analysis_result()[0]
        mean_CC_L4E, std_CC_L4E = mean_and_std(numpy.array(
            s.values)[idxs, :][:, idxs][numpy.triu_indices(sum(idxs == True), 1)])

        s = queries.param_filter_query(
            self.datastore, st_name='InternalStimulus', sheet_name='V1_Inh_L4').get_segments()[0]
        isis = [numpy.diff(st.magnitude) for st in s.spiketrains]
        idxs = numpy.array([len(isi) for isi in isis]) > 5
        mean_CV_L4I, std_CV_L4I = mean_and_std(numpy.array(
            [numpy.std(isi)/numpy.mean(isi) for isi in isis])[idxs])
        s = queries.param_filter_query(self.datastore, st_name='InternalStimulus', sheet_name='V1_Inh_L4',
                                       value_name='Correlation coefficient(psth (bin=10.0))', ads_unique=True).get_analysis_result()[0]
        mean_CC_L4I, std_CC_L4I = mean_and_std(numpy.array(
            s.values)[idxs, :][:, idxs][numpy.triu_indices(sum(idxs == True), 1)])

        s = queries.param_filter_query(
            self.datastore, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3').get_segments()[0]
        isis = [numpy.diff(st.magnitude) for st in s.spiketrains]
        idxs = numpy.array([len(isi) for isi in isis]) > 5
        mean_CV_L23E, std_CV_L23E = mean_and_std(numpy.array(
            [numpy.std(isi)/numpy.mean(isi) for isi in isis])[idxs])
        s = queries.param_filter_query(self.datastore, st_name='InternalStimulus', sheet_name='V1_Exc_L2/3',
                                       value_name='Correlation coefficient(psth (bin=10.0))', ads_unique=True).get_analysis_result()[0]
        mean_CC_L23E, std_CC_L23E = mean_and_std(numpy.array(
            s.values)[idxs, :][:, idxs][numpy.triu_indices(sum(idxs == True), 1)])

        s = queries.param_filter_query(
            self.datastore, st_name='InternalStimulus', sheet_name='V1_Inh_L2/3').get_segments()[0]
        isis = [numpy.diff(st.magnitude) for st in s.spiketrains]
        idxs = numpy.array([len(isi) for isi in isis]) > 5
        mean_CV_L23I, std_CV_L23I = mean_and_std(numpy.array(
            [numpy.std(isi)/numpy.mean(isi) for isi in isis])[idxs])
        s = queries.param_filter_query(self.datastore, st_name='InternalStimulus', sheet_name='V1_Inh_L2/3',
                                       value_name='Correlation coefficient(psth (bin=10.0))', ads_unique=True).get_analysis_result()[0]
        mean_CC_L23I, std_CC_L23I = mean_and_std(numpy.array(
            s.values)[idxs, :][:, idxs][numpy.triu_indices(sum(idxs == True), 1)])

        logger.info('mean_CV_L4E :' + str(mean_CV_L4E))
        logger.info('mean_CV_L4I :' + str(mean_CV_L4I))
        logger.info('mean_CV_L23E :' + str(mean_CV_L23E))
        logger.info('mean_CV_L23I :' + str(mean_CV_L23I))

        logger.info('mean_CC_L4E :' + str(mean_CC_L4E))
        logger.info('mean_CC_L4I :' + str(mean_CC_L4I))
        logger.info('mean_CC_L23E :' + str(mean_CC_L23E))
        logger.info('mean_CC_L23I :' + str(mean_CC_L23I))

        def ms(a): return (numpy.mean(a), numpy.std(a))
        mean_VM_L4E, std_VM_L4E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L4', st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
        mean_VM_L4I, std_VM_L4I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L4', st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
        if l23_flag:
            mean_VM_L23E, std_VM_L23E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L2/3', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
            mean_VM_L23I, std_VM_L23I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L2/3', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(VM)', ads_unique=True).get_analysis_result()[0].values)
        else:
            mean_VM_L23E, std_VM_L23E = 0, 0
            mean_VM_L23I, std_VM_L23I = 0, 0
        logger.info('mean_VM_L4E :' + str(mean_VM_L4E))
        logger.info('mean_VM_L4I :' + str(mean_VM_L4I))
        logger.info('mean_VM_L23E :' + str(mean_VM_L23E))
        logger.info('mean_VM_L23I :' + str(mean_VM_L23I))

        mean_CondE_L4E, std_CondE_L4E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
        mean_CondE_L4I, std_CondE_L4I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
        if l23_flag:
            mean_CondE_L23E, std_CondE_L23E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
            mean_CondE_L23I, std_CondE_L23I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ECond)', ads_unique=True).get_analysis_result()[0].values)
        else:
            mean_CondE_L23E, std_CondE_L23E = 0, 0
            mean_CondE_L23I, std_CondE_L23I = 0, 0

        logger.info('mean_ECond :' + str((mean_CondE_L4E+0.25 *
                                          mean_CondE_L4I+mean_CondE_L23E+0.25*mean_CondE_L23I)/2.5))

        mean_CondI_L4E, std_CondI_L4E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
        mean_CondI_L4I, std_CondI_L4I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L4', st_direct_stimulation_name=None, st_name=[
                                           'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
        if l23_flag:
            mean_CondI_L23E, std_CondI_L23E = ms(param_filter_query(self.datastore, sheet_name='V1_Exc_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
            mean_CondI_L23I, std_CondI_L23I = ms(param_filter_query(self.datastore, sheet_name='V1_Inh_L2/3', st_direct_stimulation_name=None, st_name=[
                                                 'InternalStimulus'], analysis_algorithm='Analog_MeanSTDAndFanoFactor', value_name='Mean(ICond)', ads_unique=True).get_analysis_result()[0].values)
        else:
            mean_CondI_L23E, std_CondI_L23E = 0, 0
            mean_CondI_L23I, std_CondI_L23I = 0, 0

        logger.info('mean_ICond :' + str((mean_CondI_L4E+0.25 *
                                          mean_CondI_L4I+mean_CondI_L23E+0.25*mean_CondI_L23I)/2.5))

        pylab.rc('axes', linewidth=1)

        def plot_with_log_normal_fit(values, gs1, gs2, x_label=False, y_label=""):
            valuesnz = values[numpy.nonzero(values)[0]]
            h, bin_edges = numpy.histogram(numpy.log10(
                valuesnz), range=(-2, 2), bins=20, normed=True)
            bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.0

            m = numpy.mean(numpy.log10(valuesnz))
            nm = numpy.mean(valuesnz)
            s = numpy.std(numpy.log10(valuesnz))
            #      if s == 0:
            #        s=1.0

            pylab.subplot(gs1)
            pylab.plot(numpy.logspace(-2, 2, 100), numpy.exp(-((numpy.log10(numpy.logspace(-2, 2, 100))-m)
                                                               ** 2)/(2*s*s))/(s*numpy.sqrt(2*numpy.pi)), linewidth=4, color="#666666")
            pylab.plot(numpy.power(10, bin_centers), h, 'ko', mec=None, mew=3)
            pylab.xlim(10**-2, 10**2)
            pylab.gca().set_xscale("log")
            if x_label:
                pylab.xlabel('firing rate [Hz]', fontsize=fontsize)
                pylab.xticks([0.01, 0.1, 1.0, 10, 100])
            else:
                pylab.xticks([])
            pylab.ylabel(y_label, fontsize=fontsize)
            pylab.yticks([0.0, 0.5, 1.0])
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())

            pylab.subplot(gs2)
            pylab.plot(numpy.logspace(-1, 2, 100), numpy.exp(-((numpy.log10(numpy.logspace(-1, 2, 100))-m)
                                                               ** 2)/(2*s*s))/(s*numpy.sqrt(2*numpy.pi)), linewidth=4, color="#666666")
            pylab.plot(numpy.logspace(-1, 2, 100),
                       numpy.exp(-numpy.logspace(-1, 2, 100)/nm)/nm, 'k--', linewidth=4)
            pylab.plot(numpy.power(10, bin_centers), h, 'ko', mec=None, mew=3)
            pylab.xlim(10**-1, 10**2)
            pylab.ylim(0.00001, 5.0)
            pylab.gca().set_xscale("log")
            pylab.gca().set_yscale("log")
            if x_label:
                pylab.xlabel('firing rate [Hz]', fontsize=fontsize)
                pylab.xticks([0.1, 1.0, 10, 100])
            else:
                pylab.xticks([])
            pylab.yticks([0.0001, 0.01, 1.0])
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())

        plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Exc_L4"], st_direct_stimulation_name=None, st_name=[
                                 'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[0:3, 2], gs[0:3, 3], y_label='L4e')
        plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Inh_L4"], st_direct_stimulation_name=None, st_name=[
                                 'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[3:6, 2], gs[3:6, 3], y_label='L4i')
        if l23_flag:
            plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Exc_L2/3"], st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[6:9, 2], gs[6:9, 3], y_label='L2/3e')
            plot_with_log_normal_fit(param_filter_query(self.datastore, value_name=['Firing rate'], sheet_name=["V1_Inh_L2/3"], st_direct_stimulation_name=None, st_name=[
                                     'InternalStimulus'], ads_unique=True).get_analysis_result()[0].values, gs[9:12, 2], gs[9:12, 3], x_label=True, y_label='L2/3i')

        def autolabel(rects, offset=0.25):
            # attach some text labels
            for rect in rects:
                height = rect.get_width()
                pylab.gca().text(rect.get_x() + rect.get_width() + abs(pylab.gca().get_xlim()[0] - pylab.gca().get_xlim()[1])*offset, rect.get_y()+0.012,
                                 '%.2g' % float(height),
                                 ha='center', va='bottom', fontsize=17)

        if True:
            pylab.subplot(gs[0:4, 0])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_firing_rate_L4E, mean_firing_rate_L23E], height=0.12, color='#000000', xerr=[
                            std_firing_rate_L4E, std_firing_rate_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_firing_rate_L4I, mean_firing_rate_L23I], height=0.12, color='#FFFFFF', xerr=[
                            std_firing_rate_L4I, std_firing_rate_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 8.0)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('firing rate (Hz)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.subplot(gs[4:8, 0])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CV_L4E, mean_CV_L23E], height=0.12, color='#000000', xerr=[
                            std_CV_L4E, std_CV_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CV_L4I, mean_CV_L23I], height=0.12, color='#FFFFFF', xerr=[
                            std_CV_L4I, std_CV_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 2.0)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('irregularity', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1, offset=0.37)
            autolabel(r2, offset=0.37)

            pylab.subplot(gs[8:12, 0])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CC_L4E, mean_CC_L23E], height=0.12, color='#000000', xerr=[
                            std_CC_L4E, std_CC_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CC_L4I, mean_CC_L23I], height=0.12, color='#FFFFFF', xerr=[
                            std_CC_L4I, std_CC_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 0.3)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('synchrony', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1, offset=0.6)
            autolabel(r2, offset=0.6)

            pylab.subplot(gs[0:4, 1])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [abs(mean_VM_L4E), numpy.abs(mean_VM_L23E)], height=0.12,
                            color='#000000', xerr=[std_VM_L4E, std_VM_L23E], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [abs(mean_VM_L4I), numpy.abs(mean_VM_L23I)], height=0.12,
                            color='#FFFFFF', xerr=[std_VM_L4I, std_VM_L23I], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            pylab.ylim(0, 1.0)
            pylab.xlim(40, 80)
            pylab.xticks([40, 60, 80], [-40, -60, -80])
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('membrane potential (mV)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.subplot(gs[4:8, 1])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CondE_L4E*1000, mean_CondE_L23E*1000], height=0.12, color='#000000', xerr=[
                            std_CondE_L4E*1000, std_CondE_L23E*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CondE_L4I*1000, mean_CondE_L23I*1000], height=0.12, color='#FFFFFF', xerr=[
                            std_CondE_L4I*1000, std_CondE_L23I*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 2.0)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('excitatory conductance (nS)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.subplot(gs[8:12, 1])
            r1 = pylab.barh(numpy.array([0.17, 0.67])-0.06, [mean_CondI_L4E*1000, mean_CondI_L23E*1000], height=0.12, color='#000000', xerr=[
                            std_CondI_L4E*1000, std_CondI_L23E*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            r2 = pylab.barh(numpy.array([0.33, 0.83])-0.06, [mean_CondI_L4I*1000, mean_CondI_L23I*1000], height=0.12, color='#FFFFFF', xerr=[
                            std_CondI_L4I*1000, std_CondI_L23I*1000], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            pylab.ylim(0, 1.0)
            pylab.xlim(0, 10)
            pylab.yticks([0.25, 0.75], ['L4', 'L2/3'])
            pylab.xlabel('inhibitory conductance (nS)', fontsize=fontsize)
            phf.three_tick_axis(pylab.gca().xaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(fontsize)
            phf.disable_top_right_axis(pylab.gca())
            autolabel(r1)
            autolabel(r2)

            pylab.rc('axes', linewidth=1)

        return plots



class OrientationTuningSummaryFiringRates(Plotting):
    required_parameters = ParameterSet({})

    required_parameters = ParameterSet({
        'exc_sheet_name1': str,  # the name of the sheet for which to plot
        'inh_sheet_name1': str,  # the name of the sheet for which to plot
        'exc_sheet_name2': str,  # the name of the sheet for which to plot
        'inh_sheet_name2': str,  # the name of the sheet for which to plot
    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(27, 39, subplot_spec=subplotspec,
                                              hspace=1.0, wspace=5.0)

        spike_ids1 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name=self.parameters.exc_sheet_name1).get_segments()[0].get_stored_spike_train_ids()))
        spike_ids_inh1 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name=self.parameters.inh_sheet_name1).get_segments()[0].get_stored_spike_train_ids()))
        spike_ids2 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name=self.parameters.exc_sheet_name2).get_segments()[0].get_stored_spike_train_ids()))
        spike_ids_inh2 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name=self.parameters.inh_sheet_name2).get_segments()[0].get_stored_spike_train_ids()))

        base = queries.param_filter_query(self.datastore, sheet_name=self.parameters.exc_sheet_name1, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation baseline of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids1)
        mmax = queries.param_filter_query(self.datastore, sheet_name=self.parameters.exc_sheet_name1, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation max of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids1)
        responsive_spike_ids1 = numpy.array(
            spike_ids1)[numpy.array(base)+numpy.array(mmax) > 1.0]

        base = queries.param_filter_query(self.datastore, sheet_name=self.parameters.inh_sheet_name1, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation baseline of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids_inh1)
        mmax = queries.param_filter_query(self.datastore, sheet_name=self.parameters.inh_sheet_name1, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation max of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids_inh1)
        responsive_spike_ids_inh1 = numpy.array(
            spike_ids_inh1)[numpy.array(base)+numpy.array(mmax) > 1.0]

        base = queries.param_filter_query(self.datastore, sheet_name=self.parameters.exc_sheet_name2, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation baseline of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids2)
        mmax = queries.param_filter_query(self.datastore, sheet_name=self.parameters.exc_sheet_name2, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation max of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids2)
        responsive_spike_ids2 = numpy.array(
            spike_ids2)[numpy.array(base)+numpy.array(mmax) > 1.0]

        base = queries.param_filter_query(self.datastore, sheet_name=self.parameters.inh_sheet_name2, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation baseline of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids_inh2)
        mmax = queries.param_filter_query(self.datastore, sheet_name=self.parameters.inh_sheet_name2, st_name=['FullfieldDriftingSinusoidalGrating'], st_contrast=100, value_name=[
                                          'orientation max of Firing rate'], ads_unique=True).get_analysis_result()[0].get_value_by_id(spike_ids_inh2)
        responsive_spike_ids_inh2 = numpy.array(
            spike_ids_inh2)[numpy.array(base)+numpy.array(mmax) > 1.0]

        spont_l4exc_pnv = param_filter_query(self.datastore, st_name='InternalStimulus', analysis_algorithm=[
                                             'TrialAveragedFiringRate'], value_name='Firing rate', sheet_name="V1_Exc_L4").get_analysis_result()[0]
        spont_l4inh_pnv = param_filter_query(self.datastore, st_name='InternalStimulus', analysis_algorithm=[
                                             'TrialAveragedFiringRate'], value_name='Firing rate', sheet_name="V1_Inh_L4").get_analysis_result()[0]
        spont_l23exc_pnv = param_filter_query(self.datastore, st_name='InternalStimulus', analysis_algorithm=[
                                              'TrialAveragedFiringRate'], value_name='Firing rate', sheet_name="V1_Exc_L2/3").get_analysis_result()[0]
        spont_l23inh_pnv = param_filter_query(self.datastore, st_name='InternalStimulus', analysis_algorithm=[
                                              'TrialAveragedFiringRate'], value_name='Firing rate', sheet_name="V1_Inh_L2/3").get_analysis_result()[0]

        dsv = queries.param_filter_query(self.datastore, st_name='FullfieldDriftingSinusoidalGrating', analysis_algorithm=[
                                         'TrialAveragedFiringRate'], value_name='Firing rate')
        plots['ExcORTCMeanL4'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True, 'pool': False,
                                                                     'polar': False}), spont_level_pnv=spont_l4exc_pnv), gs[0:6, :6], {'y_lim': (0, None), 'title': None, 'x_label': None, 'y_label': 'Layer 4 (EXC)\n\nfiring rate (sp/s)', 'x_ticks': None, 'linestyles': ['--', '-', '-']})
        plots['ExcORTC1L4'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids1[0:3]), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': False, 'pool': False,
                                                                  'polar': False}), spont_level_pnv=spont_l4exc_pnv), gs[0:3, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': False, 'x_axis': False, 'x_ticks': False, 'linestyles': ['--', '-', '-']})
        plots['ExcORTC2L4'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids1[3:6]), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': False,
                                                                  'pool': False, 'polar': False}), spont_level_pnv=spont_l4exc_pnv), gs[3:6, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': False, 'x_axis': False, 'linestyles': ['--', '-', '-']})

        plots['InhORTCMeanL4'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True, 'mean': True, 'pool': False,
                                                                     'polar': False}), spont_level_pnv=spont_l4inh_pnv), gs[7:13, :6], {'y_lim': (0, None), 'title': None, 'x_label': None, 'y_label': 'Layer 4 (INH)\n\nfiring rate (sp/s)', 'x_ticks': None, 'linestyles': ['--', '-', '-']})
        plots['InhORTC1L4'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids_inh1[0:3]), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True, 'mean': False,
                                                                  'pool': False, 'polar': False}), spont_level_pnv=spont_l4inh_pnv), gs[7:10, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': False, 'x_axis': False, 'linestyles': ['--', '-', '-']})
        plots['InhORTC2L4'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids_inh1[3:6]), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True, 'mean': False, 'pool': False,
                                                                  'polar': False}), spont_level_pnv=spont_l4inh_pnv), gs[10:13, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': None, 'x_axis': False, 'x_ticks': False, 'linestyles': ['--', '-', '-']})

        plots['ExcORTCMeanL23'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True, 'mean': True, 'pool': False,
                                                                      'polar': False}), spont_level_pnv=spont_l23exc_pnv), gs[14:20, :6], {'y_lim': (0, None), 'title': None, 'x_label': None, 'y_label': 'Layer 2/3 (EXC)\n\nfiring rate (sp/s)', 'x_ticks': None, 'linestyles': ['--', '-', '-']})
        plots['ExcORTC1L23'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids2[0:3]), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True, 'mean': False, 'pool': False,
                                                                   'polar': False}), spont_level_pnv=spont_l23exc_pnv), gs[14:17, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': False, 'x_axis': False, 'x_ticks': False, 'linestyles': ['--', '-', '-']})
        plots['ExcORTC2L23'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids2[3:6]), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True, 'mean': False, 'pool': False,
                                                                   'polar': False}), spont_level_pnv=spont_l23exc_pnv), gs[17:20, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': False, 'x_axis': False, 'linestyles': ['--', '-', '-']})

        plots['InhORTCMeanL23'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True, 'mean': True,
                                                                      'pool': False, 'polar': False}), spont_level_pnv=spont_l23inh_pnv), gs[21:27, :6], {'y_lim': (0, None), 'title': None, 'y_label': 'Layer 2/3 (INH)\n\nfiring rate (sp/s)', 'linestyles': ['--', '-', '-']})
        plots['InhORTC1L23'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids_inh2[0:3]), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True, 'mean': False,
                                                                   'pool': False, 'polar': False}), spont_level_pnv=spont_l23inh_pnv), gs[21:24, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'x_label': None, 'y_axis': False, 'x_axis': False, 'linestyles': ['--', '-', '-']})
        plots['InhORTC2L23'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(responsive_spike_ids_inh2[3:6]), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True, 'mean': False,
                                                                   'pool': False, 'polar': False}), spont_level_pnv=spont_l23inh_pnv), gs[24:27, 6:15], {'y_lim': (0, None), 'title': None, 'left_border': None, 'y_axis': None, 'x_axis': False, 'linestyles': ['--', '-', '-']})

        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'orientation HWHH of Firing rate'], sheet_name=[self.parameters.exc_sheet_name1])
        plots['HWHHExcL4'] = (PerNeuronValueScatterPlot(dsv, ParameterSet({'only_matching_units': True, 'ignore_nan': True})), gs[0:6, 17:23], {
                              'x_lim': (0, 50), 'y_lim': (0, 50), 'identity_line': True, 'x_label': None, 'y_label': 'HWHH cont. 5%', 'cmp': None, 'title': None})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'orientation HWHH of Firing rate'], sheet_name=[self.parameters.inh_sheet_name1])
        plots['HWHHInhL4'] = (PerNeuronValueScatterPlot(dsv, ParameterSet({'only_matching_units': True, 'ignore_nan': True})), gs[7:13, 17:23], {
                              'x_lim': (0, 50), 'y_lim': (0, 50), 'identity_line': True, 'x_label': None, 'y_label': 'HWHH cont. 5%', 'cmp': None, 'title': None})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'orientation HWHH of Firing rate'], sheet_name=[self.parameters.exc_sheet_name2])
        plots['HWHHExcL23'] = (PerNeuronValueScatterPlot(dsv, ParameterSet({'only_matching_units': True, 'ignore_nan': True})), gs[14:20, 17:23], {
                               'x_lim': (0, 50), 'y_lim': (0, 50), 'identity_line': True, 'x_label': None, 'y_label': 'HWHH cont. 5%', 'cmp': None, 'title': None})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'orientation HWHH of Firing rate'], sheet_name=[self.parameters.inh_sheet_name2])
        plots['HWHHInhL23'] = (PerNeuronValueScatterPlot(dsv, ParameterSet({'only_matching_units': True, 'ignore_nan': True})), gs[21:27, 17:23], {
                               'x_lim': (0, 50), 'y_lim': (0, 50), 'identity_line': True, 'x_label': 'HWHH Cont. 100%', 'y_label': 'HWHH cont. 5%', 'cmp': None, 'title': None})

        dsv = queries.param_filter_query(self.datastore, value_name=['orientation HWHH of Firing rate'], sheet_name=[
                                         self.parameters.exc_sheet_name1], st_contrast=[100])
        plots['HWHHHistogramExcL4'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[0:6, 26:32], {
                                       'x_lim': (0.0, 50.0), 'x_label': None, 'title': None, 'y_label': '# neurons'})
        dsv = queries.param_filter_query(self.datastore, value_name=['orientation HWHH of Firing rate'], sheet_name=[
                                         self.parameters.inh_sheet_name1], st_contrast=[100])
        plots['HWHHHistogramInhL4'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[7:13, 26:32], {
                                       'x_lim': (0.0, 50.0), 'x_label': None, 'title': None, 'y_label': '# neurons'})
        dsv = queries.param_filter_query(self.datastore, value_name=['orientation HWHH of Firing rate'], sheet_name=[
                                         self.parameters.exc_sheet_name2], st_contrast=[100])
        plots['HWHHHistogramExcL23'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[14:20, 26:32], {
                                        'x_lim': (0.0, 50.0), 'x_label': None, 'title': None, 'y_label': '# neurons'})
        dsv = queries.param_filter_query(self.datastore, value_name=['orientation HWHH of Firing rate'], sheet_name=[
                                         self.parameters.inh_sheet_name2], st_contrast=[100])
        plots['HWHHHistogramInhL23'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[21:27, 26:32], {
                                        'x_lim': (0.0, 50.0), 'x_label': 'HWHH (100% cont.)', 'title': None, 'y_label': '# neurons'})

        dsv = queries.param_filter_query(self.datastore, value_name=['orientation CV(Firing rate)'], sheet_name=[
                                         self.parameters.exc_sheet_name1], st_contrast=[100])
        plots['CVHistogramExcL4'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[0:6, 33:39], {
                                     'x_lim': (0.0, 1.0), 'x_label': None, 'title': None, 'y_label': None})
        dsv = queries.param_filter_query(self.datastore, value_name=['orientation CV(Firing rate)'], sheet_name=[
                                         self.parameters.inh_sheet_name1], st_contrast=[100])
        plots['CVHistogramInhL4'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[7:13, 33:39], {
                                     'x_lim': (0.0, 1.0), 'x_label': None, 'title': None, 'y_label': None})
        dsv = queries.param_filter_query(self.datastore, value_name=['orientation CV(Firing rate)'], sheet_name=[
                                         self.parameters.exc_sheet_name2], st_contrast=[100])
        plots['CVHistogramExcL23'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[14:20, 33:39], {
                                      'x_lim': (0.0, 1.0), 'x_label': None, 'title': None, 'y_label': None})
        dsv = queries.param_filter_query(self.datastore, value_name=['orientation CV(Firing rate)'], sheet_name=[
                                         self.parameters.inh_sheet_name2], st_contrast=[100])
        plots['CVHistogramInhL23'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[21:27, 33:39], {
                                      'x_lim': (0.0, 1.0), 'x_label': 'CV (100% cont.)', 'title': None, 'y_label': None})

        return plots


class OrientationTuningSummaryAnalogSignals(Plotting):

    required_parameters = ParameterSet({
        'exc_sheet_name1': str,  # the name of the sheet for which to plot
        'inh_sheet_name1': str,  # the name of the sheet for which to plot
        'exc_sheet_name2': str,  # the name of the sheet for which to plot
        'inh_sheet_name2': str,  # the name of the sheet for which to plot
    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(24, 35, subplot_spec=subplotspec,
                                              hspace=10.0, wspace=0.5)

        analog_ids1 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name=self.parameters.exc_sheet_name1).get_segments()[0].get_stored_esyn_ids()))
        analog_ids_inh1 = sorted(numpy.random.permutation(queries.param_filter_query(
            self.datastore, sheet_name=self.parameters.inh_sheet_name1).get_segments()[0].get_stored_esyn_ids()))
        or_tuning_exc1 = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.exc_sheet_name1)[0]
        or_tuning_inh1 = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.inh_sheet_name1)[0]

        if self.parameters.exc_sheet_name2 != 'None':
            analog_ids2 = sorted(numpy.random.permutation(queries.param_filter_query(
                self.datastore, sheet_name=self.parameters.exc_sheet_name2).get_segments()[0].get_stored_esyn_ids()))
            analog_ids_inh2 = sorted(numpy.random.permutation(queries.param_filter_query(
                self.datastore, sheet_name=self.parameters.inh_sheet_name2).get_segments()[0].get_stored_esyn_ids()))
            or_tuning_exc2 = self.datastore.get_analysis_result(
                identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.exc_sheet_name2)[0]
            or_tuning_inh2 = self.datastore.get_analysis_result(
                identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.inh_sheet_name2)[0]

        or_tuning_exc1 = None
        or_tuning_inh1 = None
        or_tuning_exc2 = None
        or_tuning_inh2 = None

        # L4 EXC
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         '-(x+y)(F0_Vm,Mean(VM))'])
        plots['L4E_F0_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True,
                                                                 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc1), gs[0:6, :5], {'title': None, 'x_label': None, 'y_label': 'Layer 4 (EXC)', 'x_axis': False, 'x_ticks': False, 'title': 'F0 of Vm (mV)'})
        dsv = queries.param_filter_query(self.datastore, value_name=['F1_Vm'])
        plots['L4E_F1_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True,
                                                                 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc1), gs[0:6, 6:11], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False, 'title': 'F1 of Vm (mV)'})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'F0_Exc_Cond-Mean(ECond)'])
        plots['L4E_F0_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True,
                                                                      'pool': False, 'polar': False}), centering_pnv=or_tuning_exc1), gs[0:6, 12:17], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False, 'title': 'F0 of gE (nS)'})
        dsv = queries.param_filter_query(
            self.datastore, value_name=['F1_Exc_Cond'])
        plots['L4E_F1_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True,
                                                                      'pool': False, 'polar': False}), centering_pnv=or_tuning_exc1), gs[0:6, 18:23], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False, 'title': 'F1 of gE (nS)'})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'F0_Inh_Cond-Mean(ICond)'])
        plots['L4E_F0_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True,
                                                                      'pool': False, 'polar': False}), centering_pnv=or_tuning_exc1), gs[0:6, 24:29], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False, 'title': 'F0 of gI (nS)'})
        dsv = queries.param_filter_query(
            self.datastore, value_name=['F1_Inh_Cond'])
        plots['L4E_F1_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids1), 'sheet_name': self.parameters.exc_sheet_name1, 'centered': True, 'mean': True,
                                                                      'pool': False, 'polar': False}), centering_pnv=or_tuning_exc1), gs[0:6, 30:35], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False, 'title': 'F1 of gI (nS)'})

        # L4 INH
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         '-(x+y)(F0_Vm,Mean(VM))'])
        plots['L4I_F0_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True,
                                                                 'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh1), gs[6:12, :5], {'title': None, 'x_label': None, 'y_label': 'Layer 4 (INH)', 'x_axis': False, 'x_ticks': False})
        dsv = queries.param_filter_query(self.datastore, value_name=['F1_Vm'])
        plots['L4I_F1_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True,
                                                                 'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh1), gs[6:12, 6:11], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'F0_Exc_Cond-Mean(ECond)'])
        plots['L4I_F0_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh1), gs[6:12, 12:17], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
        dsv = queries.param_filter_query(
            self.datastore, value_name=['F1_Exc_Cond'])
        plots['L4I_F1_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh1), gs[6:12, 18:23], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
        dsv = queries.param_filter_query(self.datastore, value_name=[
                                         'F0_Inh_Cond-Mean(ICond)'])
        plots['L4I_F0_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh1), gs[6:12, 24:29], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
        dsv = queries.param_filter_query(
            self.datastore, value_name=['F1_Inh_Cond'])
        plots['L4I_F1_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh1), 'sheet_name': self.parameters.inh_sheet_name1, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh1), gs[6:12, 30:35], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})

        if self.parameters.exc_sheet_name2 != 'None':
            # L2/3 EXC
            dsv = queries.param_filter_query(self.datastore, value_name=[
                                             '-(x+y)(F0_Vm,Mean(VM))'])
            plots['L23E_F0_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc2), gs[12:18, :5], {'title': None, 'x_label': None, 'y_label': 'Layer 2/3 (EXC)', 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(
                self.datastore, value_name=['F1_Vm'])
            plots['L23E_F1_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc2), gs[12:18, 6:11], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(self.datastore, value_name=[
                                             'F0_Exc_Cond-Mean(ECond)'])
            plots['L23E_F0_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc2), gs[12:18, 12:17], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(
                self.datastore, value_name=['F1_Exc_Cond'])
            plots['L23E_F1_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc2), gs[12:18, 18:23], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(self.datastore, value_name=[
                                             'F0_Inh_Cond-Mean(ICond)'])
            plots['L23E_F0_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc2), gs[12:18, 24:29], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(
                self.datastore, value_name=['F1_Inh_Cond'])
            plots['L23E_F1_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids2), 'sheet_name': self.parameters.exc_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_exc2), gs[12:18, 30:35], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})

            # L2/3 INH
            dsv = queries.param_filter_query(self.datastore, value_name=[
                                             '-(x+y)(F0_Vm,Mean(VM))'])
            plots['L23I_F0_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2,
                                                                      'centered': True, 'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh2), gs[18:24, :5], {'title': None, 'x_label': None, 'y_label': 'Layer 2/3 (INH)'})
            dsv = queries.param_filter_query(
                self.datastore, value_name=['F1_Vm'])
            plots['L23I_F1_Vm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True,
                                                                      'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh2), gs[18:24, 6:11], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(self.datastore, value_name=[
                                             'F0_Exc_Cond-Mean(ECond)'])
            plots['L23I_F0_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh2), gs[18:24, 12:17], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(
                self.datastore, value_name=['F1_Exc_Cond'])
            plots['L23I_F1_CondExc'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh2), gs[18:24, 18:23], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(self.datastore, value_name=[
                                             'F0_Inh_Cond-Mean(ICond)'])
            plots['L23I_F0_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh2), gs[18:24, 24:29], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})
            dsv = queries.param_filter_query(
                self.datastore, value_name=['F1_Inh_Cond'])
            plots['L23I_F1_CondInh'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'orientation', 'neurons': list(analog_ids_inh2), 'sheet_name': self.parameters.inh_sheet_name2, 'centered': True,
                                                                           'mean': True, 'pool': False, 'polar': False}), centering_pnv=or_tuning_inh2), gs[18:24, 30:35], {'title': None, 'x_label': None, 'y_label': None, 'x_axis': False, 'x_ticks': False})

        return plots


class TrialToTrialVariabilityComparison(Plotting):

    required_parameters = ParameterSet({
        'sheet_name1': str,  # The name of the sheet in which to do the analysis
        'sheet_name2': str,  # The name of the sheet in which to do the analysis
        'data_ni': float,
        'data_dg': float,
    })

    def plot(self):
        self.fig = pylab.figure(facecolor='w', **self.fig_param)
        gs = gridspec.GridSpec(1, 3)
        gs.update(left=0.07, right=0.97, top=0.9, bottom=0.1, wspace=0.1)

        orr = list(set([MozaikParametrized.idd(s).orientation for s in queries.param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_contrast=100).get_stimuli()]))
        l4_exc_or = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.sheet_name1)
        l23_exc_or = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.sheet_name2)

        # lets calculate spont. activity trial to trial variability
        # we assume that the spontaneous activity had already the spikes removed

        def calculate_sp(datastore, sheet_name):
            dsv = queries.param_filter_query(datastore, st_name='InternalStimulus', st_direct_stimulation_name=None,
                                             sheet_name=sheet_name, analysis_algorithm='ActionPotentialRemoval', ads_unique=True)
            ids = dsv.get_analysis_result()[0].ids
            sp = {}
            for idd in ids:
                assert len(dsv.get_analysis_result()) == 1
                s = dsv.get_analysis_result()[0].get_asl_by_id(idd).magnitude
                num_trials = 10
                sp[idd] = 1/numpy.mean(numpy.std([s[i*int(len(s)/num_trials):(i+1)*int(
                    len(s)/num_trials)] for i in range(0, num_trials)], axis=0, ddof=1))

            return sp

        sp_l4 = calculate_sp(self.datastore, self.parameters.sheet_name1)
        if self.parameters.sheet_name2 != 'None':
            sp_l23 = calculate_sp(self.datastore, self.parameters.sheet_name2)
        else:
            sp_l23 = 0

        def calculate_var_ratio(datastore, sheet_name, sp, ors):
            # lets calculate the mean of trial-to-trial variances across the neurons in the datastore for gratings
            dsv = queries.param_filter_query(datastore, st_name='FullfieldDriftingSinusoidalGrating', sheet_name=sheet_name,
                                             st_contrast=100, analysis_algorithm='TrialVariability', y_axis_name='Vm (no AP) trial-to-trial variance')
            assert queries.equal_ads(dsv, except_params=['stimulus_id'])
            ids = dsv.get_analysis_result()[0].ids

            std_gr = 0

            for i in ids:
                # find the or pereference of the neuron
                o = orr[numpy.argmin(
                    [circular_dist(o, ors[0].get_value_by_id(i), numpy.pi) for o in orr])]
                assert len(queries.param_filter_query(
                    dsv, st_orientation=o, ads_unique=True).get_analysis_result()) == 1
                a = 1/numpy.mean(numpy.sqrt(queries.param_filter_query(dsv, st_orientation=o,
                                                                       ads_unique=True).get_analysis_result()[0].get_asl_by_id(i).magnitude))
                std_gr = std_gr + a / sp[i]

            std_gr = std_gr / len(ids)

            # lets calculate the mean of trial-to-trial variances across the neurons in the datastore for natural images
            dsv = queries.param_filter_query(datastore, st_name='NaturalImageWithEyeMovement',
                                             sheet_name=sheet_name, y_axis_name='Vm (no AP) trial-to-trial variance', ads_unique=True)
            std_ni_ind = [1/numpy.mean(numpy.sqrt(dsv.get_analysis_result()
                                                  [0].get_asl_by_id(i).magnitude)) / sp[i] for i in ids]
            std_ni = numpy.mean(std_ni_ind)

            return std_gr, std_ni

        var_gr_l4, var_ni_l4 = calculate_var_ratio(
            self.datastore, self.parameters.sheet_name1, sp_l4, l4_exc_or)
        if self.parameters.sheet_name2 != 'None':
            var_gr_l23, var_ni_l23 = calculate_var_ratio(
                self.datastore, self.parameters.sheet_name2, sp_l23, l23_exc_or)
        else:
            var_gr_l23, var_ni_l23 = 0, 0

        lw = pylab.rcParams['axes.linewidth']
        pylab.rc('axes', linewidth=3)
        width = 0.25
        x = numpy.array([width, 1-width])

        def plt(a, b):
            rects = pylab.bar(
                x-width/2.0, [a*100-100, b*100-100], width=width, color='k')
            pylab.xlim(0, 1.0)
            pylab.ylim(-30, 70)
            pylab.xticks(x, ["DG", "NI"])
            pylab.yticks([-30, 0, 70], ["70%", "100%", "170%"])
            pylab.axhline(0.0, color='k', linewidth=3)
            disable_top_right_axis(pylab.gca())
            disable_xticks(pylab.gca())
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(19)
            rects[0].set_color('r')

        ax = pylab.subplot(gs[0, 0])
        plt(self.parameters.data_dg, self.parameters.data_ni)
        pylab.title("Data", fontsize=19, y=1.05)

        ax = pylab.subplot(gs[0, 1])
        plt(var_gr_l4, var_ni_l4)
        disable_left_axis(ax)
        remove_y_tick_labels()
        pylab.title("Layer 4", fontsize=19, y=1.05)

        ax = pylab.subplot(gs[0, 2])
        plt(var_gr_l23, var_ni_l23)
        disable_left_axis(ax)
        remove_y_tick_labels()
        pylab.title("Layer 2/3", fontsize=19, y=1.05)

        pylab.rc('axes', linewidth=lw)

        if self.plot_file_name:
            pylab.savefig(Global.root_directory+self.plot_file_name)


class TrialCrossCorrelationAnalysis(Plotting):
    """
    Trial-to-trial crosscorrelation analysis replicated from figure 4D:
    Baudot, P., Levy, M., Marre, O., Monier, C., Pananceau, M., & Frgnac, Y. (2013). Animation of natural scene by virtual eye-movements evokes high precision and low noise in V1 neurons. Frontiers in neural circuits, 7(December), 206. doi:10.3389/fncir.2013.00206

    Differences:

    Notes:
    It assumes that the TrialToTrialCrossCorrelationOfPSTHandVM analysis was run on natural images, and that it was run with the 2.0 ms  bin lentgth for calculating of PSTH
    and that the optimal preferred orientation for all the neurons for which the .
    """

    required_parameters = ParameterSet({
        'neurons1': list,  # The list of neurons to include in the analysis
        'neurons2': list,  # The list of neurons to include in the analysis
        'sheet_name1': str,
        'sheet_name2': str,
        'window_length': int,  # ms

    })

    def __init__(self, datastore, parameters, plot_file_name=None, fig_param=None, frame_duration=0):
        Plotting.__init__(self, datastore, parameters,
                          plot_file_name, fig_param, frame_duration)

    def calculate_cc(self, sheet_name, neurons):
        orr = list(set([MozaikParametrized.idd(s).orientation for s in queries.param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_contrast=100).get_stimuli()]))
        oor = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='orientation preference', sheet_name=sheet_name)

        vm_gr_asls = []
        psth_gr_asls = []

        dsv1 = queries.param_filter_query(self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_contrast=100,
                                          sheet_name=sheet_name, analysis_algorithm='TrialToTrialCrossCorrelationOfAnalogSignalList')

        if True:
            for neuron_idd in neurons:
                col = orr[numpy.argmin(
                    [circular_dist(o, oor[0].get_value_by_id(neuron_idd), numpy.pi) for o in orr])]
                logger.info("HOHOHOA: " + str(col))
                dsv = queries.param_filter_query(
                    dsv1, y_axis_name='trial-trial cross-correlation of Vm (no AP)', st_orientation=col, ads_unique=True)
                dsv.print_content()
                vm_gr_asls.append(dsv.get_analysis_result()[
                                  0].get_asl_by_id(neuron_idd))
                logger.info("HOHOHOB: " + str(len(vm_gr_asls[-1])))
                dsv = queries.param_filter_query(
                    dsv1, y_axis_name='trial-trial cross-correlation of psth (bin=10.0)', st_orientation=col, ads_unique=True)
                psth_gr_asls.append(dsv.get_analysis_result()[
                                    0].get_asl_by_id(neuron_idd))

        vm_cc_gr = numpy.mean(numpy.array(vm_gr_asls), axis=0)
        psth_cc_gr = numpy.mean(numpy.array(psth_gr_asls), axis=0)

        dsv = queries.param_filter_query(self.datastore, y_axis_name='trial-trial cross-correlation of Vm (no AP)',
                                         st_name="NaturalImageWithEyeMovement", sheet_name=sheet_name, ads_unique=True)
        vm_cc_ni = numpy.mean(numpy.array(
            dsv.get_analysis_result()[0].asl), axis=0)
        dsv = queries.param_filter_query(self.datastore, y_axis_name='trial-trial cross-correlation of psth (bin=10.0)',
                                         st_name="NaturalImageWithEyeMovement", sheet_name=sheet_name, ads_unique=True)
        psth_cc_ni = numpy.mean(numpy.array(
            dsv.get_analysis_result()[0].asl), axis=0)

        return numpy.squeeze(vm_cc_gr), numpy.squeeze(psth_cc_gr), numpy.squeeze(vm_cc_ni), numpy.squeeze(psth_cc_ni)

    def _fitgaussian(self, X, Y):
        def fitfunc(p, x): return p[0] + p[1] * numpy.exp(-numpy.abs(0-x)**2/(2*p[2]**2))
        def errfunc(p, x, y): return fitfunc(p, x) - y  # Distance to the target function

        p0 = [0, 1.0, 10, 0.0]  # Initial guess for the parameters
        p0[0] = numpy.min(Y)
        p0[1] = numpy.max(Y)-p0[0]
        p0[3] = 0  # ,numpy.average(numpy.array(X),weights=numpy.array(Y))

        p1, success = scipy.optimize.leastsq(errfunc, numpy.array(p0[:]), args=(numpy.array(X), numpy.array(Y)))
        p1[2] = abs(p1[2])

        if success:
            return p1
        else:
            return [0, 0, 0, 0]

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=subplotspec)

        vm_cc_gr_s1, psth_cc_gr_s1, vm_cc_ni_s1, psth_cc_ni_s1 = self.calculate_cc(
            self.parameters.sheet_name1, self.parameters.neurons1)
        vm_cc_gr_s2, psth_cc_gr_s2, vm_cc_ni_s2, psth_cc_ni_s2 = self.calculate_cc(
            self.parameters.sheet_name2, self.parameters.neurons2)

        vm_cc_gr_pool, psth_cc_gr_pool, vm_cc_ni_pool, psth_cc_ni_pool = (
            vm_cc_gr_s1+vm_cc_gr_s2)/2, (psth_cc_gr_s1+psth_cc_gr_s2)/2, (vm_cc_ni_s1+vm_cc_ni_s2)/2, (psth_cc_ni_s1+psth_cc_ni_s2)/2

        z = int(min(self.parameters.window_length,
                    (len(vm_cc_gr_s1)-1)/2, (len(vm_cc_gr_s2)-1)/2)/2)*2

        a = 0.6

        p0, p1, p2, p3 = self._fitgaussian(numpy.linspace(-z, z, z+1), psth_cc_gr_s1[int(
            len(psth_cc_gr_s1)/2)-int(z/2):int(len(psth_cc_gr_s1)/2)+int(z/2)+1])
        p0, p1, p2, p3 = self._fitgaussian(numpy.linspace(-z, z, z+1), psth_cc_ni_s1[int(
            len(psth_cc_ni_s1)/2)-int(z/2):int(len(psth_cc_ni_s1)/2)+int(z/2)+1])
        p0, p1, p2, p3 = self._fitgaussian(numpy.linspace(-z, z, z+1), psth_cc_gr_s2[int(
            len(psth_cc_gr_s2)/2)-int(z/2):int(len(psth_cc_gr_s2)/2)+int(z/2)+1])
        p0, p1, p2, p3 = self._fitgaussian(numpy.linspace(-z, z, z+1), psth_cc_ni_s2[int(
            len(psth_cc_ni_s2)/2)-int(z/2):int(len(psth_cc_ni_s2)/2)+int(z/2)+1])

        p0, p1, p2, p3 = self._fitgaussian(
            numpy.linspace(-z, z, 2*z+1), vm_cc_gr_s1[int(len(vm_cc_gr_s1)/2)-z:int(len(vm_cc_gr_s1)/2)+z+1])
        p0, p1, p2, p3 = self._fitgaussian(
            numpy.linspace(-z, z, 2*z+1), vm_cc_ni_s1[int(len(vm_cc_ni_s1)/2)-z:int(len(vm_cc_ni_s1)/2)+z+1])
        p0, p1, p2, p3 = self._fitgaussian(
            numpy.linspace(-z, z, 2*z+1), vm_cc_gr_s2[int(len(vm_cc_gr_s2)/2)-z:int(len(vm_cc_gr_s2)/2)+z+1])
        p0, p1, p2, p3 = self._fitgaussian(
            numpy.linspace(-z, z, 2*z+1), vm_cc_ni_s2[int(len(vm_cc_ni_s2)/2)-z:int(len(vm_cc_ni_s2)/2)+z+1])

        plots["Spike_sheet_1"] = (StandardStyleLinePlot([numpy.linspace(-z, z, z+1), numpy.linspace(-z, z, z+1)], [psth_cc_gr_s1[int(len(psth_cc_gr_s1)/2)-int(z/2):int(len(psth_cc_gr_s1)/2)+int(z/2)+1], psth_cc_ni_s1[int(len(psth_cc_ni_s1)/2)-int(z/2):int(len(
            psth_cc_ni_s1)/2)+int(z/2+1)]]), gs[0, 0], {'colors': ['r', 'k'], 'x_tick_style': 'Custom', 'x_ticks': [], 'y_tick_style': 'Custom', 'y_ticks': [0, 0.2], 'y_tick_labels': [0.0, 0.2], 'linewidth': 2.0, 'y_lim': (-0.02, 0.2), 'y_label': 'spikes'})
        plots["Vm_sheet_1"] = (StandardStyleLinePlot([numpy.linspace(-z, z, 2*z+1), numpy.linspace(-z, z, 2*z+1)], [vm_cc_gr_s1[int(len(vm_cc_gr_s1)/2)-z:int(len(vm_cc_gr_s1)/2)+z+1], vm_cc_ni_s1[int(len(vm_cc_ni_s1)/2)-z:int(len(vm_cc_ni_s1)/2)+z+1]]), gs[1, 0], {'x_label': 'time(ms)', 'colors': [
                               'r', 'k'], 'x_tick_style': 'Custom', 'x_ticks': [-z, 0, z], 'x_tick_labels': [-self.parameters.window_length, 0, self.parameters.window_length], 'y_tick_style': 'Custom', 'y_ticks': [-a, 0, a], 'y_tick_labels': [-a, 0.0, a], 'linewidth': 2.0, 'y_lim': (-a, a), 'y_label': 'Vm'})
        plots["Spike_sheet_2"] = (StandardStyleLinePlot([numpy.linspace(-z, z, z+1), numpy.linspace(-z, z, z+1)], [psth_cc_gr_s2[int(len(psth_cc_gr_s2)/2)-int(z/2):int(len(psth_cc_gr_s2)/2)+int(z/2)+1], psth_cc_ni_s2[int(len(psth_cc_ni_s2)/2)-int(z/2):int(len(psth_cc_ni_s2)/2)+int(z/2+1)]]),
                                  gs[0, 1], {'colors': ['r', 'k'], 'x_tick_style': 'Custom', 'x_ticks': [], 'y_tick_style': 'Custom', 'y_ticks': [0, 0.2], 'y_tick_labels': [0.0, 0.2], 'linewidth': 2.0, 'y_lim': (-0.02, 0.2), 'y_label': 'spikes', 'y_ticks': None, 'y_label': None})
        plots["Vm_sheet_2"] = (StandardStyleLinePlot([numpy.linspace(-z, z, 2*z+1), numpy.linspace(-z, z, 2*z+1)], [vm_cc_gr_s2[int(len(vm_cc_gr_s2)/2)-z:int(len(vm_cc_gr_s2)/2)+z+1], vm_cc_ni_s2[int(len(vm_cc_ni_s2)/2)-z:int(len(vm_cc_ni_s2)/2)+z+1]]), gs[1, 1], {'x_label': 'time(ms)', 'colors': [
                               'r', 'k'], 'x_tick_style': 'Custom', 'x_ticks': [-z, 0, z], 'x_tick_labels': [-self.parameters.window_length, 0, self.parameters.window_length], 'y_tick_style': 'Custom', 'y_ticks': [-a, 0, a], 'y_tick_labels': [-a, 0.0, a], 'linewidth': 2.0, 'y_lim': (-a, a), 'y_label': 'Vm', 'y_ticks': None, 'y_label': None})
        plots["Spike_sheet_pool"] = (StandardStyleLinePlot([numpy.linspace(-z, z, z+1), numpy.linspace(-z, z, z+1)], [psth_cc_gr_pool[int(len(psth_cc_gr_pool)/2)-int(z/2):int(len(psth_cc_gr_pool)/2)+int(z/2)+1], psth_cc_ni_pool[int(len(psth_cc_ni_pool)/2)-int(z/2):int(len(psth_cc_ni_pool)/2)+int(z/2)+1]]),
                                     gs[0, 2], {'colors': ['r', 'k'], 'x_tick_style': 'Custom', 'x_ticks': [], 'y_tick_style': 'Custom', 'y_ticks': [0.0, 0.2], 'y_tick_labels': [0.0, 0.2], 'linewidth': 2.0, 'y_lim': (-0.02, 0.2), 'y_label': 'spikes', 'y_ticks': None, 'y_label': None})
        plots["Vm_sheet_pool"] = (StandardStyleLinePlot([numpy.linspace(-z, z, 2*z+1), numpy.linspace(-z, z, 2*z+1)], [vm_cc_gr_pool[int(len(vm_cc_gr_pool)/2)-z:int(len(vm_cc_gr_pool)/2)+z+1], vm_cc_ni_pool[int(len(vm_cc_ni_pool)/2)-z:int(len(vm_cc_ni_pool)/2)+z+1]]), gs[1, 2], {'x_label': 'time(ms)', 'colors': [
                                  'r', 'k'], 'x_tick_style': 'Custom', 'x_ticks': [-z, 0, z], 'x_tick_labels': [-self.parameters.window_length, 0, self.parameters.window_length], 'y_tick_style': 'Custom', 'y_ticks': [-a, 0, a], 'y_tick_labels': [-a, 0.0, a], 'linewidth': 2.0, 'y_lim': (-a, a), 'y_label': 'Vm', 'y_ticks': None, 'y_label': None})

        return plots


class SizeTuningOverview(Plotting):
    required_parameters = ParameterSet({
        'l4_neurons': list,
        'l23_neurons': list,
        'l4_neurons_analog': list,
        'l23_neurons_analog': list,
    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            8, 24, subplot_spec=subplotspec, hspace=1.0, wspace=0.3)
        fontsize = 20

        low_contrast = str(30)

        dsv = param_filter_query(
            self.datastore, st_name='DriftingSinusoidalGratingDisk', value_name=['Firing rate'])
        plots['L4ExcFR'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': self.parameters.l4_neurons, 'sheet_name': 'V1_Exc_L4', 'centered': False, 'mean': True, 'polar': False, 'pool': False})), gs[0:4, 0:4], {
                            'fontsize': fontsize, 'title': None, 'x_label': None, 'y_label': r'Firing rate ($\frac{sp}{s}$)', 'y_lim': (0, 8), 'x_axis': False, 'x_ticks': False, 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})
        if self.parameters.l23_neurons != []:
            plots['L23ExcFR'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': self.parameters.l23_neurons, 'sheet_name': 'V1_Exc_L2/3', 'centered': False, 'mean': True, 'polar': False, 'pool': False})), gs[4:8, 0:4], {
                                 'fontsize': fontsize, 'title': None, 'y_label': r'Firing rate ($\frac{sp}{s}$)', 'y_lim': (0, 8), 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})
        # (x+y)(F1_Vm,-(x+y)(F0_Vm,Mean(VM)))
        dsv = param_filter_query(self.datastore, st_name='DriftingSinusoidalGratingDisk', value_name=[
                                 '-(x+y)(F0_Vm,Mean(VM))'])
        plots['L4ExcVm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': self.parameters.l4_neurons_analog, 'sheet_name': 'V1_Exc_L4', 'centered': False, 'mean': True, 'polar': False, 'pool': False})), gs[0:4, 5:9], {
                            'fontsize': fontsize, 'title': None, 'x_label': None, 'y_label': r'Vm (mV)', 'x_axis': False, 'x_ticks': False, 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})
        if self.parameters.l23_neurons != []:
            dsv = param_filter_query(self.datastore, st_name='DriftingSinusoidalGratingDisk', value_name=[
                                     '-(x+y)(F0_Vm,Mean(VM))'])
            plots['L23ExcVm'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': self.parameters.l23_neurons_analog, 'sheet_name': 'V1_Exc_L2/3', 'centered': False, 'mean': True,
                                                                    'polar': False, 'pool': False})), gs[4:8, 5:9], {'fontsize': fontsize, 'title': None, 'y_label': r'Vm (mV)', 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})

        dsv = param_filter_query(self.datastore, value_name=[
                                 'Suppression index of Firing rate'], sheet_name='V1_Exc_L4')
        plots['L4ExcSI'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[0:4, 10:14], {'fontsize': fontsize, 'title': None, 'x_label': None, 'y_label': '# neurons',
                                                                                                              'x_axis': False, 'x_ticks': False, 'num_bins': 10, 'mark_mean': True, 'x_lim': (0, 1.0), 'y_lim': (0, 20), 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})
        if self.parameters.l23_neurons != []:
            dsv = param_filter_query(self.datastore, value_name=[
                                     'Suppression index of Firing rate'], sheet_name='V1_Exc_L2/3')
            plots['L2/3ExcSI'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[4:8, 10:14], {'fontsize': fontsize, 'title': None, 'x_label': None, 'y_label': '# neurons',
                                                                                                                    'x_label': 'Suppression index', 'num_bins': 10, 'mark_mean': True, 'x_lim': (0, 1.0), 'y_lim': (0, 20), 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})

        dsv = param_filter_query(self.datastore, value_name=[
                                 'Max. facilitation radius of Firing rate'], sheet_name='V1_Exc_L4')
        plots['L4ExcMaxFacilitationRadius'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[0:4, 15:19], {'fontsize': fontsize, 'title': None, 'x_label': None, 'y_label': '# neurons',
                                                                                                                                 'x_axis': False, 'x_ticks': False, 'num_bins': 8, 'mark_mean': True, 'x_lim': (0, 4.0), 'y_lim': (0, 20), 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})
        if self.parameters.l23_neurons != []:
            dsv = param_filter_query(self.datastore, value_name=[
                                     'Max. facilitation radius of Firing rate'], sheet_name='V1_Exc_L2/3')
            plots['L2/3ExcMaxFacilitationRadius'] = (PerNeuronValuePlot(dsv, ParameterSet({'cortical_view': False})), gs[4:8, 15:19], {'fontsize': fontsize, 'title': None, 'x_label': None, 'y_label': '# neurons',
                                                                                                                                       'x_label': 'Maximum facillitation radius', 'num_bins': 8, 'mark_mean': True, 'x_lim': (0, 4.0), 'y_lim': (0, 20), 'colors': {'contrast : 100': '#000000', 'contrast : ' + low_contrast: '#0073B3'}})

        dsv = param_filter_query(self.datastore, st_name='DriftingSinusoidalGratingDisk', value_name=[
                                 'F1_Exc_Cond', 'F1_Inh_Cond'])
        #dsv = param_filter_query(self.datastore,st_name='DriftingSinusoidalGratingDisk',value_name=['(x+y)(F0_Exc_Cond,F1_Exc_Cond)','(x+y)(F0_Inh_Cond,F1_Inh_Cond)'])
        plots['L4ExcCond,'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': self.parameters.l4_neurons_analog, 'sheet_name': 'V1_Exc_L4', 'centered': False, 'mean': True, 'polar': False, 'pool': True})), gs[0:4, 20:24], {
                               'fontsize': fontsize, 'title': None, 'x_label': None, 'x_axis': False, 'x_ticks': False, 'colors': {'F1_Exc_Cond contrast : 100': '#FF0000', 'F1_Exc_Cond contrast : ' + low_contrast: '#FFACAC', 'F1_Inh_Cond contrast : 100': '#0000FF', 'F1_Inh_Cond contrast : ' + low_contrast: '#ACACFF'}, 'y_label': 'Conductance (nS)'})
        if self.parameters.l23_neurons != []:
            dsv = param_filter_query(self.datastore, st_name='DriftingSinusoidalGratingDisk', value_name=[
                                     'F0_Exc_Cond', 'F0_Inh_Cond'])
            #dsv = param_filter_query(self.datastore,st_name='DriftingSinusoidalGratingDisk',value_name=['(x+y)(F0_Exc_Cond,F1_Exc_Cond)','(x+y)(F0_Inh_Cond,F1_Inh_Cond)'])
            plots['L23ExcCond'] = (PlotTuningCurve(dsv, ParameterSet({'parameter_name': 'radius', 'neurons': self.parameters.l23_neurons_analog, 'sheet_name': 'V1_Exc_L2/3', 'centered': False, 'mean': True, 'polar': False, 'pool': True})), gs[4:8, 20:24], {
                                   'fontsize': fontsize, 'title': None, 'colors': {'F0_Exc_Cond contrast : 100': '#FF0000', 'F0_Exc_Cond contrast : ' + low_contrast: '#FFACAC', 'F0_Inh_Cond contrast : 100': '#0000FF', 'F0_Inh_Cond contrast : ' + low_contrast: '#ACACFF'}, 'y_label': 'Conductance (nS)'})

        return plots




class StimulusResponseComparison(Plotting):
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'neuron': int,  # which neuron to show
    })

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(1, 21, subplot_spec=subplotspec,
                                              hspace=1.0, wspace=1.0)

        orr = list(set([MozaikParametrized.idd(s).orientation for s in queries.param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_contrast=100).get_stimuli()]))
        #ors = self.datastore.get_analysis_result(identifier='PerNeuronValue',value_name = 'LGNAfferentOrientation', sheet_name = self.parameters.sheet_name)

        #dsv = queries.param_filter_query(self.datastore,st_name='FullfieldDriftingSinusoidalGrating',st_orientation=orr[numpy.argmin([circular_dist(o,ors[0].get_value_by_id(self.parameters.neuron),numpy.pi)  for o in orr])],st_contrast=100)
        dsv = queries.param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_orientation=0, st_contrast=100)
        plots['Gratings'] = (OverviewPlot(dsv, ParameterSet(
            {'sheet_name': self.parameters.sheet_name, 'neuron': self.parameters.neuron, 'spontaneous': True, 'sheet_activity': {}})), gs[:, :10], {})
        #dsv = queries.param_filter_query(self.datastore,st_name='DriftingGratingWithEyeMovement')
        #plots['GratingsWithEM'] = (OverviewPlot(dsv, ParameterSet({'sheet_name': self.parameters.sheet_name,'neuron': self.parameters.neuron, 'spontaneous' : True,'sheet_activity' : {}})),gs[2:4,:],{'x_label': None})
        dsv = queries.param_filter_query(
            self.datastore, st_name='NaturalImageWithEyeMovement')
        plots['NIwEM'] = (OverviewPlot(dsv, ParameterSet(
            {'sheet_name': self.parameters.sheet_name, 'neuron': self.parameters.neuron, 'spontaneous': True, 'sheet_activity': {}})), gs[:, 11:], {'y_label': None})

        return plots


class SizeTuningOverviewNew(Plotting):
    """
    The analysis of size tuning in the model.


    Parameters
    ----------
    l4_neurons : list
           The list of layer 4 neurons to include in the analysis.

    l23_neurons : list
           The list of layer 23 neurons to include in the analysis.

    l4_neurons_analog : list
           The list of layer 4 neurons for which sub-threshold signals were recorded to include in the analysis.

    l23_neurons_analog : list
           The list of layer 23 neurons for which sub-threshold signals were recorded to include in the analysis.
    """

    required_parameters = ParameterSet({
        'l4_neurons': list,
        'l23_neurons': list,
        'l4_neurons_analog': list,
        'l23_neurons_analog': list,
    })

    err = 0

    def _fitgaussian(self, X, Y):
        from scipy.special import erf

        #fitfunc = lambda p,x:  p[0]*erf(x/p[1])**2 - p[0]*p[2] *erf(x/(p[1] + p[3]))**2 + p[0]*p[4]*p[2] *erf(x/(p[1]+ p[3]+p[5]))**2 + p[6]
        def fitfunc(p, x): return p[0]*erf(x/p[1])**2 - p[2] * erf(
            x/(p[1] + p[3]))**2 + p[4] * erf(x/(p[1] + p[3]+p[5]))**2 + p[6]
        def errfunc(p, x, y): return numpy.linalg.norm(
            fitfunc(p, x) - y)  # Distance to the target function

        err = []
        res = []
        # Initial guess for the parameters
        p0 = [8.0, 0.43, 8.0, 0.18, 3.0, 1.4, numpy.min(Y)]

        for i in range(2, 15):
            for j in range(5, 11):
                for k in range(1, 5):
                    p0[1] = i/15.0
                    p0[3] = j/10.0
                    p0[5] = k/2.0
                r = scipy.optimize.fmin_tnc(errfunc, numpy.array(p0), args=(numpy.array(X), numpy.array(Y)), bounds=[
                                            (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)], approx_grad=True)
                res.append(r)
                err.append(errfunc(r[0], numpy.array(X), numpy.array(Y)))

        res = res[numpy.argmin(err)]
        self.err = self.err+numpy.min(err)

        x = numpy.linspace(0, 3.0, 100)

        # pylab.figure()
        # pylab.plot(x,fitfunc(res[0],x),'-')
        # pylab.hold('on')
        # pylab.plot(X,Y,'x')
        # pylab.title(str(res[0]))

        return fitfunc(res[0], x)

    def get_vals(self, dsv, neuron):
        assert queries.ads_with_equal_stimulus_type(dsv)
        assert queries.equal_ads(dsv, except_params=['stimulus_id'])
        pnvs = dsv.get_analysis_result()

        st = [MozaikParametrized.idd(s.stimulus_id) for s in pnvs]
        tc_dict = colapse_to_dictionary(
            [z.get_value_by_id(neuron) for z in pnvs], st, "radius")

        rads = tc_dict.values()[0][0]*2
        values = tc_dict.values()[0][1]
        a, b = zip(*sorted(zip(rads, values)))
        return numpy.array(a), numpy.array(b)

    def subplot(self, subplotspec):

        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            26, 30, subplot_spec=subplotspec, hspace=2.0, wspace=1.2)
        fontsize = 19
        low_contrast = 30

        # NICE L4 NEURON
        def example_neuron(neuron, line, sheet):

            rads_lc, values_lc = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue', sheet_name=sheet,
                                                                          st_name='DriftingSinusoidalGratingDisk', value_name='Firing rate', analysis_algorithm='TrialAveragedFiringRate', st_contrast=low_contrast), neuron)
            fitvalues_lc = self._fitgaussian(rads_lc, values_lc)

            rads_hc, values_hc = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue', sheet_name=sheet,
                                                                          st_name='DriftingSinusoidalGratingDisk', value_name='Firing rate', analysis_algorithm='TrialAveragedFiringRate', st_contrast=100), neuron)
            fitvalues_hc = self._fitgaussian(rads_hc, values_hc)

            ax = pylab.subplot(gs[6*line:6*line+6, 1:9])
            ax.plot(rads_lc*2, values_lc, 'ok')
            ax.plot(numpy.linspace(0, 6.0, 100), fitvalues_lc, 'k')
            ax.plot(rads_hc*2, values_hc, 'o', color='#0073B3',
                    markeredgecolor='#0073B3', markeredgewidth=0)
            ax.plot(numpy.linspace(0, 6.0, 100), fitvalues_hc, color='#0073B3')
            if line == 2:
                ax.set_ylim(0, 3.0)
            else:
                ax.set_ylim(0, 5.2)
            disable_top_right_axis(pylab.gca())
            three_tick_axis(ax.yaxis)
            if line == 2:
                # three_tick_axis(ax.xaxis)
                ax.set_xticks([0, 3, 6.0])
            else:
                remove_x_tick_labels()
                disable_xticks(ax)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(19)
            pylab.ylabel('firing rate (sp/s)', fontsize=fontsize)

            var = 'F1_Vm'
            # if line==2:
            #  var = '-(x+y)(F0_Vm,Mean(VM))'

            rads_lc, values_lc = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                          sheet_name=sheet, st_name='DriftingSinusoidalGratingDisk', value_name=[var], st_contrast=low_contrast), neuron)
            fitvalues_lc = self._fitgaussian(rads_lc, values_lc)

            rads_hc, values_hc = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                          sheet_name=sheet, st_name='DriftingSinusoidalGratingDisk', value_name=[var], st_contrast=100), neuron)
            fitvalues_hc = self._fitgaussian(rads_hc, values_hc)

            ax = pylab.subplot(gs[6*line:6*line+6, 11:19])
            ax.plot(rads_lc*2, values_lc, 'ok')
            ax.plot(numpy.linspace(0, 6.0, 100), fitvalues_lc, 'k')
            ax.plot(rads_hc*2, values_hc, 'o', color='#0073B3',
                    markeredgecolor='#0073B3', markeredgewidth=0)
            ax.plot(numpy.linspace(0, 6.0, 100), fitvalues_hc, color='#0073B3')
            if line == 2:
                ax.set_ylim(0, 1.8)
            elif line == 1:
                ax.set_ylim(0, 5.8)
            elif line == 0:
                ax.set_ylim(0, 4.5)

            disable_top_right_axis(pylab.gca())
            three_tick_axis(ax.yaxis)
            if line == 2:
                # three_tick_axis(ax.xaxis)
                ax.set_xticks([0, 3, 6.0])
            else:
                disable_xticks(ax)
                remove_x_tick_labels()

            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(19)
            pylab.ylabel('Vm (mV)', fontsize=fontsize)

            rads_lc_e, values_lc_e = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue', sheet_name=sheet,
                                                                              st_name='DriftingSinusoidalGratingDisk', value_name=['x-y(F0_Exc_Cond,Mean(ECond))'], st_contrast=low_contrast), neuron)
            values_lc_e *= 1000
            fitvalues_lc_e = self._fitgaussian(rads_lc_e, values_lc_e)

            rads_hc_e, values_hc_e = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue', sheet_name=sheet,
                                                                              st_name='DriftingSinusoidalGratingDisk', value_name=['x-y(F0_Exc_Cond,Mean(ECond))'], st_contrast=100), neuron)
            values_hc_e *= 1000
            fitvalues_hc_e = self._fitgaussian(rads_hc_e, values_hc_e)

            rads_lc_i, values_lc_i = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue', sheet_name=sheet,
                                                                              st_name='DriftingSinusoidalGratingDisk', value_name=['x-y(F0_Inh_Cond,Mean(ICond))'], st_contrast=low_contrast), neuron)
            values_lc_i *= 1000
            fitvalues_lc_i = self._fitgaussian(rads_lc_i, values_lc_i)

            rads_hc_i, values_hc_i = self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue', sheet_name=sheet,
                                                                              st_name='DriftingSinusoidalGratingDisk', value_name=['x-y(F0_Inh_Cond,Mean(ICond))'], st_contrast=100), neuron)
            values_hc_i *= 1000
            fitvalues_hc_i = self._fitgaussian(rads_hc_i, values_hc_i)

            ax = pylab.subplot(gs[6*line:6*line+6, 21:29])
            ax.plot(rads_lc_e*2, values_lc_e, 'o', color='#FF0000',
                    markeredgecolor='#FF0000', markeredgewidth=0)
            ax.plot(numpy.linspace(0, 6.0, 100),
                    fitvalues_lc_e, color='#FF0000')
            ax.plot(rads_hc_e*2, values_hc_e, 'o', color='#FFACAC',
                    markeredgecolor='#FFACAC', markeredgewidth=0)
            ax.plot(numpy.linspace(0, 6.0, 100),
                    fitvalues_hc_e, color='#FFACAC')

            ax.plot(rads_lc_i*2, values_lc_i, 'o', color='#0000FF',
                    markeredgecolor='#0000FF', markeredgewidth=0)
            ax.plot(numpy.linspace(0, 6.0, 100),
                    fitvalues_lc_i, color='#0000FF')
            ax.plot(rads_hc_i*2, values_hc_i, 'o', color='#ACACFF',
                    markeredgecolor='#ACACFF', markeredgewidth=0)
            ax.plot(numpy.linspace(0, 6.0, 100),
                    fitvalues_hc_i, color='#ACACFF')
            ax.set_ylim(0, 24)

            disable_top_right_axis(pylab.gca())
            if line == 2:
                # three_tick_axis(ax.xaxis)
                ax.set_xticks([0, 3, 6.0])
            else:
                remove_x_tick_labels()
                disable_xticks(ax)

            three_tick_axis(ax.yaxis)
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(19)
            pylab.ylabel('conductance (nS)', fontsize=fontsize)

        nice_neuron_l4 = self.parameters.l4_neurons_analog[0]  # 25432
        not_nice_neuron_l4 = self.parameters.l4_neurons_analog[0]  # 34816
        nice_neuron_l23 = self.parameters.l23_neurons_analog[0]  # 60674

        example_neuron(nice_neuron_l4, 0, 'V1_Exc_L4')
        example_neuron(not_nice_neuron_l4, 1, 'V1_Exc_L4')
        example_neuron(nice_neuron_l23, 2, 'V1_Exc_L2/3')

        def size_tuning_measures(rads, values):
            crf_index = numpy.argmax(values[:-1]-values[1:] > 0)
            if crf_index == 0:
                crf_index = len(values)-1

            crf_size = rads[crf_index]

            if crf_index < len(values)-1 and crf_index != 0:
                supp_index = crf_index+numpy.argmin(values[crf_index+1:])+1
            else:
                supp_index = len(values)-1

            if supp_index < len(values)-1 and supp_index != 0:
                cs_index = supp_index+numpy.argmax(values[supp_index+1:])+1
            else:
                cs_index = len(values)-1

            if values[crf_index] != 0:
                si = (values[crf_index]-values[supp_index])/values[crf_index]
            else:
                si = 0

            if values[crf_index] != 0:
                csi = (values[cs_index]-values[supp_index])/values[crf_index]
            else:
                csi = 0
            return [crf_size, si, csi]

        selected_l4_neurons = [neuron for neuron in self.parameters.l4_neurons if numpy.max(self.get_vals(queries.param_filter_query(
            self.datastore, identifier='PerNeuronValue', sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', st_contrast=100), neuron)) > 2.0]
        selected_l23_neurons = [neuron for neuron in self.parameters.l23_neurons if numpy.max(self.get_vals(queries.param_filter_query(
            self.datastore, identifier='PerNeuronValue', sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', st_contrast=100), neuron)) > 2.0]

        l4_hc_crf_size, l4_hc_si, l4_hc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 6.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                  sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', st_contrast=100), neuron))) for neuron in selected_l4_neurons])
        l4_lc_crf_size, l4_lc_si, l4_lc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 6.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                  sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', st_contrast=low_contrast), neuron))) for neuron in selected_l4_neurons])
        l23_hc_crf_size, l23_hc_si, l23_hc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 6.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                     sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', st_contrast=100), neuron))) for neuron in selected_l23_neurons])
        l23_lc_crf_size, l23_lc_si, l23_lc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 6.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                     sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', analysis_algorithm='TrialAveragedFiringRate', value_name='Firing rate', st_contrast=low_contrast), neuron))) for neuron in selected_l23_neurons])
        ax = pylab.subplot(gs[19:24, 1:6])
        ax.plot(l4_hc_si, l4_lc_si, 'ow', markeredgecolor='k')
        ax.plot(l23_hc_si, l23_lc_si, 'ok')
        ax.plot([0, 1], [0, 1], 'k')
        disable_top_right_axis(pylab.gca())
        three_tick_axis(ax.yaxis)
        three_tick_axis(ax.xaxis)
        pylab.xlim(0, 0.6)
        pylab.ylim(0, 0.6)
        # pylab.title('spikes',fontsize=fontsize)

        ax.annotate("", xy=(numpy.mean(l4_hc_si+l23_hc_si), 0.55), xycoords='data', xytext=(numpy.mean(l4_hc_si+l23_hc_si),
                                                                                            0.6), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))
        ax.annotate("", xy=(0.55, numpy.mean(l4_lc_si+l23_lc_si)), xycoords='data', xytext=(0.6, numpy.mean(l4_lc_si+l23_lc_si)),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))

        for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
            label.set_fontsize(19)
        pylab.xlabel('SI (high-contrast)', fontsize=fontsize)
        pylab.ylabel('SI (low-contrast)', fontsize=fontsize)

        def mean_and_sem(x): return (numpy.mean(
            x), numpy.std(x)/numpy.sqrt(len(x)))

        print('SI (high-contrast): L4 ', str(mean_and_sem(l4_hc_si)))
        print('SI (high-contrast): L23 ', str(mean_and_sem(l23_hc_si)))
        print('SI (low-contrast): L4', str(mean_and_sem(l4_lc_si)))
        print('SI (low-contrast): L23', str(mean_and_sem(l23_lc_si)))

        ax = pylab.subplot(gs[19:24, 7:12])
        ax.plot(l4_hc_csi, l4_lc_csi, 'ow', markeredgecolor='k')
        ax.plot(l23_hc_csi, l23_lc_csi, 'ok')
        ax.plot([0, 1], [0, 1], 'k')
        disable_top_right_axis(pylab.gca())
        three_tick_axis(ax.yaxis)
        three_tick_axis(ax.xaxis)
        pylab.xlim(0, 0.6)
        pylab.ylim(0, 0.6)
        # pylab.title('spikes',fontsize=fontsize)

        ax.annotate("", xy=(numpy.mean(l4_hc_csi+l23_hc_csi), 0.55), xycoords='data', xytext=(numpy.mean(l4_hc_csi+l23_hc_csi),
                                                                                              0.6), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))
        ax.annotate("", xy=(0.55, numpy.mean(l4_lc_csi+l23_lc_csi)), xycoords='data', xytext=(0.6, numpy.mean(l4_lc_csi+l23_lc_csi)),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))

        for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
            label.set_fontsize(19)
        pylab.xlabel('CSI (high-contrast)', fontsize=fontsize)
        pylab.ylabel('CSI (low-contrast)', fontsize=fontsize)
        print('CSI (high-contrast): L4 ', str(mean_and_sem(l4_hc_csi)))
        print('CSI (high-contrast): L23 ', str(mean_and_sem(l23_hc_csi)))
        print('CSI (low-contrast): L4', str(mean_and_sem(l4_lc_csi)))
        print('CSI (low-contrast): L23', str(mean_and_sem(l23_lc_csi)))

        ax = pylab.subplot(gs[19:24, 13:18])
        ax.plot(l4_hc_si, l4_hc_csi, 'ow', markeredgecolor='k')
        ax.plot(l23_hc_si, l23_hc_csi, 'ok')
        ax.plot([0, 1], [0, 1], 'k')
        disable_top_right_axis(pylab.gca())
        three_tick_axis(ax.yaxis)
        three_tick_axis(ax.xaxis)
        pylab.xlim(0, 0.6)
        pylab.ylim(0, 0.6)
        # pylab.title('spikes',fontsize=fontsize)

        for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
            label.set_fontsize(19)
        pylab.xlabel('SI (high-contrast)', fontsize=fontsize)
        pylab.ylabel('CSI (high-contrast)', fontsize=fontsize)

        ax.annotate("", xy=(numpy.mean(l4_hc_si+l23_hc_si), 0.55), xycoords='data', xytext=(numpy.mean(l4_hc_si+l23_hc_si),
                                                                                            0.6), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))
        ax.annotate("", xy=(0.55, numpy.mean(l4_hc_csi+l23_hc_csi)), xycoords='data', xytext=(0.6, numpy.mean(l4_hc_csi+l23_hc_csi)),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))

        ax = pylab.subplot(gs[19:24, 19:24])
        ax.plot(l4_hc_crf_size, l4_lc_crf_size, 'ow', markeredgecolor='k')
        ax.plot(l23_hc_crf_size, l23_lc_crf_size, 'ok')
        ax.plot([0, 6], [0, 6], 'k')
        disable_top_right_axis(pylab.gca())
        three_tick_axis(ax.yaxis)
        three_tick_axis(ax.xaxis)
        pylab.xlim(0, 6.0)
        pylab.ylim(0, 6.0)
        #pylab.title('membrane potential',fontsize=fontsize)

        ax.annotate("", xy=(numpy.mean(l4_hc_crf_size+l23_hc_crf_size), 5.7), xycoords='data', xytext=(numpy.mean(l4_hc_crf_size +
                                                                                                                  l23_hc_crf_size), 6.0), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))
        ax.annotate("", xy=(5.7, numpy.mean(l4_lc_crf_size+l23_lc_crf_size)), xycoords='data', xytext=(6.0, numpy.mean(l4_lc_crf_size +
                                                                                                                       l23_lc_crf_size)), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='k'))

        for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
            label.set_fontsize(19)
        pylab.xlabel('CRF size (high-contrast)', fontsize=fontsize)
        pylab.ylabel('CRF size (low-contrast)', fontsize=fontsize)

        print('MFR (high-contrast): L4 ', str(mean_and_sem(l4_hc_crf_size)))
        print('MFR (high-contrast): L23 ', str(mean_and_sem(l23_hc_crf_size)))
        print('MFR (low-contrast): L4', str(mean_and_sem(l4_lc_crf_size)))
        print('MFR (low-contrast): L23', str(mean_and_sem(l23_lc_crf_size)))

        l4_hc_crf_size, l4_hc_si, l4_hc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                  sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Exc_Cond,Mean(ECond))', st_contrast=100), neuron))) for neuron in self.parameters.l4_neurons_analog])
        l4_lc_crf_size, l4_lc_si, l4_lc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                  sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Exc_Cond,Mean(ECond))', st_contrast=low_contrast), neuron))) for neuron in self.parameters.l4_neurons_analog])
        l23_hc_crf_size, l23_hc_si, l23_hc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                     sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Exc_Cond,Mean(ECond))', st_contrast=100), neuron))) for neuron in self.parameters.l23_neurons_analog])
        l23_lc_crf_size, l23_lc_si, l23_lc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                     sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Exc_Cond,Mean(ECond))', st_contrast=low_contrast), neuron))) for neuron in self.parameters.l23_neurons_analog])

        ax = pylab.subplot(gs[19:24, 25:30])
        ax.plot(l4_hc_si, l4_lc_si, 'ow', markeredgecolor='#FF0000')
        ax.plot(l23_hc_si, l23_lc_si, 'o',
                color='#FF0000', markeredgecolor='#FF0000')

        ax.annotate("", xy=(numpy.mean(l4_hc_si+l23_hc_si), 0.55), xycoords='data', xytext=(numpy.mean(l4_hc_si+l23_hc_si), 0.6),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='#FF0000'))
        ax.annotate("", xy=(0.55, numpy.mean(l4_lc_si+l23_lc_si)), xycoords='data', xytext=(0.6, numpy.mean(l4_lc_si+l23_lc_si)),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='#FF0000'))

        l4_hc_crf_size, l4_hc_si, l4_hc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                  sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Inh_Cond,Mean(ICond))', st_contrast=100), neuron))) for neuron in self.parameters.l4_neurons_analog])
        l4_lc_crf_size, l4_lc_si, l4_lc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                  sheet_name='V1_Exc_L4', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Inh_Cond,Mean(ICond))', st_contrast=low_contrast), neuron))) for neuron in self.parameters.l4_neurons_analog])
        l23_hc_crf_size, l23_hc_si, l23_hc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                     sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Inh_Cond,Mean(ICond))', st_contrast=100), neuron))) for neuron in self.parameters.l23_neurons_analog])
        l23_lc_crf_size, l23_lc_si, l23_lc_csi = zip(*[size_tuning_measures(numpy.linspace(0, 3.0, 100), self._fitgaussian(*self.get_vals(queries.param_filter_query(self.datastore, identifier='PerNeuronValue',
                                                                                                                                                                     sheet_name='V1_Exc_L2/3', st_name='DriftingSinusoidalGratingDisk', value_name='x-y(F0_Inh_Cond,Mean(ICond))', st_contrast=low_contrast), neuron))) for neuron in self.parameters.l23_neurons_analog])

        ax.plot(l4_hc_si, l4_lc_si, 'ow', markeredgecolor='#0000FF')
        ax.plot(l23_hc_si, l23_lc_si, 'o',
                color='#0000FF', markeredgecolor='#0000FF')

        ax.annotate("", xy=(numpy.mean(l4_hc_si+l23_hc_si), 0.55), xycoords='data', xytext=(numpy.mean(l4_hc_si+l23_hc_si), 0.6),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='#0000FF'))
        ax.annotate("", xy=(0.55, numpy.mean(l4_lc_si+l23_lc_si)), xycoords='data', xytext=(0.6, numpy.mean(l4_lc_si+l23_lc_si)),
                    textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=3.0, color='#0000FF'))

        ax.plot([0, 1], [0, 1], 'k')
        disable_top_right_axis(pylab.gca())
        three_tick_axis(ax.yaxis)
        three_tick_axis(ax.xaxis)
        pylab.xlim(0, 0.6)
        pylab.ylim(0, 0.6)
        #pylab.title('syn. conductances',fontsize=fontsize)

        for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
            label.set_fontsize(19)
        pylab.xlabel('SI (high-contrast)', fontsize=fontsize)
        pylab.ylabel('SI (low-contrast)', fontsize=fontsize)

        return {}


class TrialToTrialVariabilityComparisonNew(Plotting):
    """
    This figure plots compares the trial-to-trial variability measure in the model to the one published in Baudot et al. 2016. 

    Parameters
    ----------
    sheet_name1 : str
               The name of the sheet corresponding to layer 4 excitatory neurons.

    sheet_name2 : str
          The name of the sheet corresponding to layer 2/3 excitatory neurons.


    data_ni : str
               The corresponding statistic for natural images in Baudot et al. 2016.

    data_dg : str
          The corresponding statistic for drifting gratings in Baudot et al. 2016.

    """

    required_parameters = ParameterSet({
        'sheet_name1': str,  # The name of the sheet in which to do the analysis
        'sheet_name2': str,  # The name of the sheet in which to do the analysis
        'data_ni': float,
        'data_dg': float,
    })

    def plot(self):
        self.fig = pylab.figure(facecolor='w', **self.fig_param)
        gs = gridspec.GridSpec(1, 3)
        gs.update(left=0.07, right=0.97, top=0.9, bottom=0.1, wspace=0.1)

        orr = list(set([MozaikParametrized.idd(s).orientation for s in queries.param_filter_query(
            self.datastore, st_name='FullfieldDriftingSinusoidalGrating', st_contrast=100).get_stimuli()]))
        l4_exc_or = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.sheet_name1)
        l23_exc_or = self.datastore.get_analysis_result(
            identifier='PerNeuronValue', value_name='LGNAfferentOrientation', sheet_name=self.parameters.sheet_name2)

        def mean_and_sem(x): return (numpy.mean(
            x), numpy.std(x)/numpy.sqrt(len(x)))
        # lets calculate spont. activity trial to trial variability
        # we assume that the spontaneous activity had already the spikes removed

        def calculate_sp(datastore, sheet_name):
            dsv = queries.param_filter_query(datastore, st_name='InternalStimulus', st_direct_stimulation_name=None,
                                             sheet_name=sheet_name, analysis_algorithm='ActionPotentialRemoval', ads_unique=True)
            ids = dsv.get_analysis_result()[0].ids
            sp = {}
            for idd in ids:
                assert len(dsv.get_analysis_result()) == 1
                s = dsv.get_analysis_result()[0].get_asl_by_id(idd).magnitude
                z = [s[i*int(len(s)/10):(i+1)*int(len(s)/10)] for i in range(0, 10)]
                sp[idd] = 1/numpy.mean(numpy.std(z, axis=0, ddof=1))

            return sp

        sp_l4 = calculate_sp(self.datastore, self.parameters.sheet_name1)
        if self.parameters.sheet_name2 != 'None':
            sp_l23 = calculate_sp(self.datastore, self.parameters.sheet_name2)
        else:
            sp_l23 = 0

        def calculate_var_ratio(datastore, sheet_name, sp, ors):
            # lets calculate the mean of trial-to-trial variances across the neurons in the datastore for gratings
            dsv = queries.param_filter_query(datastore, st_name='FullfieldDriftingSinusoidalGrating',
                                             sheet_name=sheet_name, st_contrast=100, analysis_algorithm='ActionPotentialRemoval')
            assert queries.equal_ads(dsv, except_params=['stimulus_id'])
            ids = dsv.get_analysis_result()[0].ids

            std_gr = []

            for i in ids:
                # find the or pereference of the neuron
                o = orr[numpy.argmin(
                    [circular_dist(o, ors[0].get_value_by_id(i), numpy.pi) for o in orr])]
                assert len(queries.param_filter_query(
                    dsv, st_orientation=o).get_analysis_result()) == 10

                s = [d.get_asl_by_id(i).magnitude[200:]
                     for d in dsv.get_analysis_result()]
                a = 1/numpy.mean(numpy.std(s, axis=0, ddof=1))

                std_gr.append(a / sp[i])

            std_gr, sem_gr = mean_and_sem(std_gr)

            # lets calculate the mean of trial-to-trial variances across the neurons in the datastore for natural images
            dsv = queries.param_filter_query(datastore, st_name='NaturalImageWithEyeMovement',
                                             sheet_name=sheet_name, analysis_algorithm='ActionPotentialRemoval')
            s = [1/numpy.mean(numpy.std([d.get_asl_by_id(
                idd).magnitude for d in dsv.get_analysis_result()], axis=0, ddof=1))/sp[idd] for idd in ids]
            std_ni, sem_ni = mean_and_sem(s)

            return std_gr, std_ni, sem_gr, sem_ni

        var_gr_l4, var_ni_l4, sem_gr_l4, sem_ni_l4 = calculate_var_ratio(
            self.datastore, self.parameters.sheet_name1, sp_l4, l4_exc_or)
        if self.parameters.sheet_name2 != 'None':
            var_gr_l23, var_ni_l23, sem_gr_l23, sem_ni_l23 = calculate_var_ratio(
                self.datastore, self.parameters.sheet_name2, sp_l23, l23_exc_or)
        else:
            var_gr_l23, var_ni_l23, sem_gr_l23, sem_ni_l23 = 0, 0, 0, 0

        lw = pylab.rcParams['axes.linewidth']
        pylab.rc('axes', linewidth=3)
        width = 0.25
        x = numpy.array([width, 1-width])

        def plt(a, b):
            rects = pylab.bar(x, [a*100-100, b*100-100],
                              width=width, color='k')
            pylab.xlim(0, 1.0)
            pylab.ylim(-20, 50)
            pylab.xticks(x, ["DG", "NI"])
            pylab.yticks([-20, 0, 50], ["80%", "100%", "150%"])
            pylab.axhline(0.0, color='k', linewidth=3)
            disable_top_right_axis(pylab.gca())
            disable_xticks(pylab.gca())
            for label in pylab.gca().get_xticklabels() + pylab.gca().get_yticklabels():
                label.set_fontsize(19)
            rects[0].set_color('r')

        ax = pylab.subplot(gs[0, 0])
        plt(self.parameters.data_dg, self.parameters.data_ni)
        pylab.title("Data", fontsize=19, y=1.05)

        ax = pylab.subplot(gs[0, 1])
        plt(var_gr_l4, var_ni_l4)
        disable_left_axis(ax)
        remove_y_tick_labels()
        pylab.title("Layer 4", fontsize=19, y=1.05)

        ax = pylab.subplot(gs[0, 2])
        plt(var_gr_l23, var_ni_l23)
        disable_left_axis(ax)
        remove_y_tick_labels()
        pylab.title("Layer 2/3", fontsize=19, y=1.05)

        pylab.rc('axes', linewidth=lw)

        if self.plot_file_name:
            pylab.savefig(Global.root_directory+self.plot_file_name)
