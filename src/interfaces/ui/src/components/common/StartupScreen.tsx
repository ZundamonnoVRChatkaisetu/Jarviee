import React, { useEffect, useState } from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import { arcReactorGlow, radialExpand, holographicFlicker } from '../../styles/animations';

const StartupContainer = styled(Box)({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: '#121212',
  zIndex: 9999,
  overflow: 'hidden',
});

const ArcReactorContainer = styled(motion.div)({
  position: 'relative',
  width: 200,
  height: 200,
  marginBottom: 40,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
});

const ArcReactorRing = styled(motion.div)(({ theme }) => ({
  position: 'absolute',
  borderRadius: '50%',
  border: `2px solid ${theme.palette.primary.main}`,
}));

const ArcReactorCore = styled(motion.div)(({ theme }) => ({
  position: 'absolute',
  width: '20%',
  height: '20%',
  borderRadius: '50%',
  backgroundColor: theme.palette.primary.main,
  boxShadow: `0 0 20px ${theme.palette.primary.main}, 0 0 40px ${theme.palette.primary.main}`,
  zIndex: 2,
}));

const LogoText = styled(motion.div)(({ theme }) => ({
  fontFamily: 'var(--font-primary)',
  fontWeight: 700,
  fontSize: '2.5rem',
  letterSpacing: 8,
  marginBottom: 20,
  color: theme.palette.primary.main,
  textShadow: `0 0 10px ${theme.palette.primary.main}`,
}));

const ProgressBarContainer = styled(Box)({
  width: '60%',
  maxWidth: 400,
  marginBottom: 40,
});

const StatusText = styled(motion.div)(({ theme }) => ({
  fontFamily: 'var(--font-code)',
  fontSize: '0.875rem',
  color: theme.palette.secondary.main,
  marginBottom: 8,
  textAlign: 'left',
  width: '60%',
  maxWidth: 400,
}));

const GridOverlay = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundImage: 'linear-gradient(rgba(100,181,246,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(100,181,246,0.05) 1px, transparent 1px)',
  backgroundSize: '40px 40px',
  opacity: 0.3,
  zIndex: 1,
});

const startupSteps = [
  { text: "Initializing system...", delay: 500 },
  { text: "Loading core modules...", delay: 1000 },
  { text: "Establishing API connections...", delay: 1500 },
  { text: "Loading knowledge base...", delay: 2000 },
  { text: "Calibrating AI systems...", delay: 2500 },
  { text: "Integration framework ready...", delay: 3000 },
  { text: "All systems operational.", delay: 3500 },
];

const StartupScreen: React.FC = () => {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  
  // Progress animation
  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prevProgress) => {
        const nextProgress = prevProgress + 1;
        if (nextProgress >= 100) {
          clearInterval(timer);
        }
        return nextProgress;
      });
    }, 35);

    return () => {
      clearInterval(timer);
    };
  }, []);
  
  // Step animation
  useEffect(() => {
    const setupStepTimer = (step: number) => {
      if (step < startupSteps.length) {
        const timer = setTimeout(() => {
          setCurrentStep(step);
          setupStepTimer(step + 1);
        }, startupSteps[step].delay);
        
        return () => clearTimeout(timer);
      }
    };
    
    setupStepTimer(0);
  }, []);
  
  return (
    <StartupContainer>
      <GridOverlay 
        component={motion.div}
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.3 }}
        transition={{ duration: 2 }}
      />
      
      <ArcReactorContainer
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1, delay: 0.5 }}
      >
        {/* Outer rings */}
        {[100, 80, 60, 40].map((size, index) => (
          <ArcReactorRing 
            key={index}
            style={{ 
              width: `${size}%`, 
              height: `${size}%`, 
            }}
            initial={{ opacity: 0, rotate: 45 }}
            animate={{ 
              opacity: 1, 
              rotate: index % 2 === 0 ? 45 : -45,
            }}
            transition={{ 
              duration: 1, 
              delay: 0.8 + (index * 0.2) 
            }}
          />
        ))}
        
        {/* Energy beams */}
        {[0, 45, 90, 135].map((angle) => (
          <Box
            key={angle}
            component={motion.div}
            sx={{
              position: 'absolute',
              width: '80%',
              height: '2px',
              background: (theme) => `linear-gradient(90deg, ${theme.palette.primary.main}00, ${theme.palette.primary.main})`,
              transform: `rotate(${angle}deg)`,
              transformOrigin: 'center',
              zIndex: 1,
            }}
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 0.7, scaleX: 1 }}
            transition={{ 
              duration: 0.8, 
              delay: 1.5
            }}
          />
        ))}
        
        {/* Core */}
        <ArcReactorCore 
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 1.3 }}
          css={`
            animation: ${arcReactorGlow} 2s infinite ease-in-out;
          `}
        />
      </ArcReactorContainer>
      
      <LogoText
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 1.8 }}
        css={`
          animation: ${holographicFlicker} 10s infinite linear;
        `}
      >
        JARVIEE
      </LogoText>
      
      <AnimatePresence mode="wait">
        <StatusText
          key={currentStep}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          {startupSteps[currentStep]?.text || ''}
        </StatusText>
      </AnimatePresence>
      
      <ProgressBarContainer>
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          color="secondary"
          sx={{ 
            height: 4, 
            borderRadius: 2,
            '& .MuiLinearProgress-bar': {
              borderRadius: 2,
            }
          }}
        />
      </ProgressBarContainer>
      
      <Typography 
        variant="caption" 
        color="text.secondary"
        component={motion.div}
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.7 }}
        transition={{ duration: 1, delay: 2 }}
      >
        © Jarviee AI System v0.1.0
      </Typography>
    </StartupContainer>
  );
};

export default StartupScreen;
