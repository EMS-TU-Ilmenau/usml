import numpy as np
import fastmat as fm
import usmltransform as us


class FourierSubSample(fm.Product):
    def __init__(self, **options):
        '''
        Initialize Fourier Subsampling Class

        Parameters
        ----------

        **options:
            See the list of special options below.

        Options
        -------
        numSamples : int
            number of frequency bins collected per a-scan
        strategy : string
            selection strategy of the fourier samples. 1) random: select
            numSamples many samples from the nZ1 possible ones from a
            uniform distribution. 2) max: use the spectrum of the pulse to
            select the numSamples many frequency bins carrying the most energy
            3) energy: use the spectrum of the pulse as a density to
            randomly select numSamples Fourier samples
        nZ1 : int
            Ultrasonic model parameter: number of timebins
        nZ2 : int
            Ultrasonic model parameter: number of depth bins
        nX : int
            Ultrasonic model parameter: number of x-axis bins
        nY : int
            Ultrasonic model parameter: number of y-axis bins
        pulseLength : int
            Ultrasonic model parameter:
        dx : float
            Ultrasonic model parameter:
        dy : float
            Ultrasonic model parameter:
        dz : float
            Ultrasonic model parameter:
        centerFreq : float
            Ultrasonic model parameter: pulse center freqency
        bandWidth : float
            Ultrasonic model parameter: pulse bandwidth
        speedOfSound : float
            Ultrasonic model parameter: speed of sound in medium
        samplingFreq : float
            Ultrasonic model parameter: sampling frequency of pulse
        foreRunLength : float
            Ultrasonic model parameter:
        beamAngle : float
            Ultrasonic model parameter: tangens of transducer beam angle
        '''

        # generate the forward and backward model implemented in cuda
        self._H = us.CudaBlockTwoLevelToeplitz(
            nZ1=options['nZ1'],
            nZ2=options['nZ2'],
            nX=options['nX'],
            nY=options['nY'],
            pulseLength=options['pulseLength'],
            dx=options['dx'],
            dy=options['dy'],
            dz=options['dz'],
            centerFreq=options['centerFreq'],
            bandWidth=options['bandWidth'],
            speedOfSound=options['speedOfSound'],
            samplingFreq=options['samplingFreq'],
            foreRunLength=options['foreRunLength'],
            beamAngle=options['beamAngle']
        )

        # size of the 2-level toeplitz blocks
        subSelSize = options['nX'] * options['nY']

        numSubSetZ = int(np.floor((options['nZ1'] + 1) / 2))
        print(numSubSetZ)

        # generate the subselection indices in the fourier domain
        if options['strategy'] == 'energy':
            # use the energy density in the frequency bins as density
            # for the subselection probabilities
            pulseSpec = np.abs(np.fft.fft(self._H.getPulse()))
            self._spec = np.copy(pulseSpec)
            pulseSpec[numSubSetZ:] = 0
            self._J = np.sort(np.random.choice(
                options['nZ1'],
                options['numSamples'],
                replace=False,
                p=pulseSpec / np.sum(pulseSpec)
            ))
        elif options['strategy'] == 'random':
            # do a uniform random sampling
            self._J = np.sort(np.random.choice(
                numSubSetZ, options['numSamples'], replace=False
            ))
        elif options['strategy'] == 'max':
            # use the bins containting the most energy
            pulseSpec = np.abs(np.fft.fft(self._H.getPulse()))
            pulseSpec[numSubSetZ:] = 0
            self._J = np.sort(
                np.argsort(pulseSpec)[-options['numSamples']:]
            )
        else:
            raise NotImplementedError("Strategy not Implemented.")

        # now make J symmetric
        self._J = np.sort(np.block([self._J, options['nZ1'] - 1 - self._J]))

        # now calculate the subselection of the large kronecker product
        self._subSel = np.empty(
            self._J.shape[0] * subSelSize,
            dtype='int64'
        )
        for ii in range(self._J.shape[0]):
            self._subSel[ii * subSelSize:(ii+1) * subSelSize] = (
                self._J[ii] * subSelSize + np.arange(subSelSize)
            )

        # create the right fourier matrix
        self._F = fm.Fourier(options['nZ1'])

        # create the kronecker product which realizes the fourier sampling
        # of the a-scans
        self._M = fm.Kron(self._F, fm.Eye(subSelSize))

        # now subselect the right rows in the kronecker product
        self._Phi = fm.Partial(self._M, rows=self._subSel)

        # now create the product
        super(FourierSubSample, self).__init__(self._Phi, self._H)

    def _getColNorms(self):
        vecPulse = self._H.getPulse()
        vecPulseHat = np.fft.fft(vecPulse)

        vecZ = np.arange(
                vecPulse.shape[0]
            ) * self._H._dz + self._H._foreRunLength

        arrNorms = np.sqrt(self._H._nZ1) * np.repeat(
            np.linalg.norm(vecPulseHat[self._J]) * (
                (2.0 * np.pi) / (
                    (
                        0.5 * self._H._beamAngle * vecZ
                    ) ** 2)
            ),
            self._H._nX * self._H._nY
        )

        return arrNorms
