  *+??΍?@sh??|?[@2?
_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::FlatMap[0]::TFRecord?m?B<??!\Ζ6܈U@)?m?B<??1\Ζ6܈U@:Advanced file read2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch??X????!R?u@)??X????1R?u@:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch?辜ٞ?!??[??@)?辜ٞ?1??[??@:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::FlatMap?:??????!????DV@)???'???1????~@:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake??_?ǣ?!
6|??L@)?????k??1 ?9?*???:Preprocessing2F
Iterator::Model{?V??ר?!B??p=@))z?c??t?1???`??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism6?ڋh;??!K?8;?@)N?f??s?1???#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.