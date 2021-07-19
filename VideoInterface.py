import abc


class VideoInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def open_video_stream(self):
        '''Opens a video stream'''
        raise NotImplementedError('Define open_video_stream to use this base class')

    @abc.abstractmethod
    def read(self, stream: object):
        '''Gets a frame from the stream'''
        raise NotImplementedError('Define read to use this base class')
