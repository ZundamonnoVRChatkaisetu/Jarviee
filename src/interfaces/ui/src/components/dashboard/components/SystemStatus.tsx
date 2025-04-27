import React from 'react';
import { 
  Box, 
  Typography, 
  LinearProgress, 
  Chip, 
  Grid,
  useTheme
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import { useSelector } from 'react-redux';
import { RootState } from '../../../store';

const StatusContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1),
}));

const StatusItem = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(2),
}));

const ProgressLabel = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: theme.spacing(0.5),
}));

const StatusGrid = styled(Grid)(({ theme }) => ({
  marginTop: theme.spacing(2),
}));

const ValueDisplay = styled(Typography)(({ theme }) => ({
  textAlign: 'center',
  fontSize: '1.5rem',
  fontWeight: 500,
  color: theme.palette.primary.main,
}));

const LabelDisplay = styled(Typography)(({ theme }) => ({
  textAlign: 'center',
  fontSize: '0.75rem',
  color: theme.palette.text.secondary,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
}));

// Custom progress bar that changes color based on value
const ColoredLinearProgress: React.FC<{
  value: number;
  warningThreshold?: number;
  criticalThreshold?: number;
}> = ({ value, warningThreshold = 60, criticalThreshold = 85 }) => {
  const theme = useTheme();
  
  let color = 'success';
  if (value >= criticalThreshold) {
    color = 'error';
  } else if (value >= warningThreshold) {
    color = 'warning';
  }
  
  return (
    <LinearProgress
      variant="determinate"
      value={value}
      color={color as 'success' | 'warning' | 'error'}
      sx={{
        height: 8,
        borderRadius: 4,
        '& .MuiLinearProgress-bar': {
          borderRadius: 4,
        },
      }}
    />
  );
};

const SystemStatus: React.FC = () => {
  const theme = useTheme();
  const { status, resourceUsage, activeModules } = useSelector(
    (state: RootState) => state.system
  );
  
  // Mock data for demonstration
  const statusData = {
    cpu: resourceUsage.cpu || 42,
    memory: resourceUsage.memory || 36,
    storage: resourceUsage.storage || 58,
    network: 12,
    uptime: '3h 24m',
    activeModules: activeModules.length || 8,
    totalModules: 12,
    systemStatus: status || 'ready',
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready':
        return <CheckCircleIcon fontSize="small" color="success" />;
      case 'initializing':
        return <WarningIcon fontSize="small" color="warning" />;
      case 'error':
        return <ErrorIcon fontSize="small" color="error" />;
      default:
        return <CheckCircleIcon fontSize="small" color="success" />;
    }
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready':
        return 'success';
      case 'initializing':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'success';
    }
  };
  
  return (
    <StatusContainer>
      {/* System Status Indicator */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          mb: 2,
        }}
      >
        <Chip
          icon={getStatusIcon(statusData.systemStatus)}
          label={`System ${statusData.systemStatus}`}
          color={getStatusColor(statusData.systemStatus) as 'success' | 'warning' | 'error'}
          variant="outlined"
          sx={{ borderWidth: 2, px: 1 }}
        />
      </Box>
      
      {/* Resource Usage */}
      <StatusItem>
        <ProgressLabel>
          <Typography variant="body2">CPU Usage</Typography>
          <Typography variant="body2" color="primary">
            {statusData.cpu}%
          </Typography>
        </ProgressLabel>
        <ColoredLinearProgress value={statusData.cpu} />
      </StatusItem>
      
      <StatusItem>
        <ProgressLabel>
          <Typography variant="body2">Memory</Typography>
          <Typography variant="body2" color="primary">
            {statusData.memory}%
          </Typography>
        </ProgressLabel>
        <ColoredLinearProgress value={statusData.memory} />
      </StatusItem>
      
      <StatusItem>
        <ProgressLabel>
          <Typography variant="body2">Storage</Typography>
          <Typography variant="body2" color="primary">
            {statusData.storage}%
          </Typography>
        </ProgressLabel>
        <ColoredLinearProgress value={statusData.storage} />
      </StatusItem>
      
      <StatusItem>
        <ProgressLabel>
          <Typography variant="body2">Network Load</Typography>
          <Typography variant="body2" color="primary">
            {statusData.network}%
          </Typography>
        </ProgressLabel>
        <ColoredLinearProgress value={statusData.network} />
      </StatusItem>
      
      {/* Additional Statistics */}
      <StatusGrid container spacing={2}>
        <Grid item xs={6}>
          <Box
            component={motion.div}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            sx={{
              p: 1.5,
              backgroundColor:
                theme.palette.mode === 'dark'
                  ? 'rgba(0, 0, 0, 0.2)'
                  : 'rgba(0, 0, 0, 0.05)',
              borderRadius: 2,
              height: '100%',
            }}
          >
            <ValueDisplay variant="h4">
              {statusData.activeModules}/{statusData.totalModules}
            </ValueDisplay>
            <LabelDisplay variant="caption">Active Modules</LabelDisplay>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box
            component={motion.div}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            sx={{
              p: 1.5,
              backgroundColor:
                theme.palette.mode === 'dark'
                  ? 'rgba(0, 0, 0, 0.2)'
                  : 'rgba(0, 0, 0, 0.05)',
              borderRadius: 2,
              height: '100%',
            }}
          >
            <ValueDisplay variant="h4">{statusData.uptime}</ValueDisplay>
            <LabelDisplay variant="caption">Uptime</LabelDisplay>
          </Box>
        </Grid>
      </StatusGrid>
    </StatusContainer>
  );
};

export default SystemStatus;
