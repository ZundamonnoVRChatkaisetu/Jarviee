import React from 'react';
import { Box, Typography, Button, Container } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import HomeIcon from '@mui/icons-material/Home';

const NotFoundContainer = styled(Container)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 'calc(100vh - var(--header-height))',
  textAlign: 'center',
  padding: theme.spacing(4),
}));

const ErrorCode = styled(Typography)(({ theme }) => ({
  fontSize: '8rem',
  fontWeight: 700,
  backgroundImage: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
  backgroundClip: 'text',
  WebkitBackgroundClip: 'text',
  color: 'transparent',
  textShadow: `0 0 15px ${theme.palette.primary.main}33`,
  lineHeight: 1,
  marginBottom: theme.spacing(2),
}));

const ErrorMessage = styled(Typography)(({ theme }) => ({
  fontSize: '1.5rem',
  marginBottom: theme.spacing(4),
  maxWidth: 500,
}));

const IconContainer = styled(motion.div)(({ theme }) => ({
  marginBottom: theme.spacing(4),
  width: 100,
  height: 100,
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: `rgba(${theme.palette.mode === 'dark' ? '255, 255, 255' : '0, 0, 0'}, 0.05)`,
}));

const NotFound: React.FC = () => {
  const navigate = useNavigate();
  
  const handleGoHome = () => {
    navigate('/dashboard');
  };
  
  return (
    <NotFoundContainer maxWidth="lg">
      <IconContainer
        initial={{ rotate: 180, scale: 0 }}
        animate={{ rotate: 0, scale: 1 }}
        transition={{
          type: 'spring',
          stiffness: 260,
          damping: 20,
        }}
      >
        <ErrorOutlineIcon 
          sx={{ 
            fontSize: 60, 
            color: 'primary.main' 
          }} 
        />
      </IconContainer>
      
      <ErrorCode 
        variant="h1"
        component={motion.h1}
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        404
      </ErrorCode>
      
      <ErrorMessage 
        variant="h2"
        component={motion.h2}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        The resource you are looking for does not exist
      </ErrorMessage>
      
      <Box
        component={motion.div}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<HomeIcon />}
          onClick={handleGoHome}
          sx={{ 
            px: 4, 
            py: 1.5, 
            borderRadius: 2,
            boxShadow: (theme) => `0 0 15px ${theme.palette.primary.main}33`,
          }}
        >
          Return to Dashboard
        </Button>
      </Box>
      
      <Typography 
        variant="body2" 
        color="text.secondary"
        sx={{ mt: 6 }}
        component={motion.p}
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.7 }}
        transition={{ delay: 0.9 }}
      >
        System message: Path not found in navigation database
      </Typography>
    </NotFoundContainer>
  );
};

export default NotFound;
