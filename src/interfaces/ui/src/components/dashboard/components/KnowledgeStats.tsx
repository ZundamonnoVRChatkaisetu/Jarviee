import React from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  LinearProgress, 
  Tooltip,
  useTheme
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import StorageIcon from '@mui/icons-material/Storage';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import PsychologyIcon from '@mui/icons-material/Psychology';

const StatsContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1),
}));

const StatItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(1.5),
}));

const IconContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 32,
  height: 32,
  borderRadius: '50%',
  marginRight: theme.spacing(1.5),
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(100, 181, 246, 0.1)' 
    : 'rgba(25, 118, 210, 0.05)',
}));

const ProgressLabel = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  width: '100%',
  marginBottom: theme.spacing(0.5),
}));

const TopicContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(0, 0, 0, 0.2)' 
    : 'rgba(0, 0, 0, 0.03)',
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(0.75, 1.5),
  marginBottom: theme.spacing(1),
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}));

// Mock data for demonstration
const knowledgeStats = {
  totalEntries: 24859,
  growthRate: 12.5,
  freshness: 87,
  confidenceScore: 92,
  topTopics: [
    { name: 'Python', entries: 5243 },
    { name: 'TypeScript', entries: 4128 },
    { name: 'React', entries: 3715 },
    { name: 'Machine Learning', entries: 2954 },
    { name: 'API Development', entries: 2219 },
  ],
};

const formatNumber = (num: number): string => {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
};

const KnowledgeStats: React.FC = () => {
  const theme = useTheme();
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.2,
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 10, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  };
  
  const topicVariants = {
    hidden: { x: -10, opacity: 0 },
    visible: {
      x: 0,
      opacity: 1,
      transition: { duration: 0.3 }
    }
  };
  
  // Calculate the percentage of entries for each topic
  const totalTopicEntries = knowledgeStats.topTopics.reduce(
    (sum, topic) => sum + topic.entries,
    0
  );
  
  return (
    <StatsContainer
      component={motion.div}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <Grid container spacing={2} mb={2}>
        <Grid item xs={6}>
          <motion.div variants={itemVariants}>
            <StatItem>
              <IconContainer>
                <StorageIcon color="primary" fontSize="small" />
              </IconContainer>
              <Box>
                <Typography variant="body2" fontWeight={500}>
                  Knowledge Base
                </Typography>
                <Typography 
                  variant="h6" 
                  color="primary" 
                  sx={{ 
                    fontWeight: 500,
                    fontSize: '1.1rem',
                    lineHeight: 1.2
                  }}
                >
                  {formatNumber(knowledgeStats.totalEntries)}
                </Typography>
                <Typography 
                  variant="caption" 
                  color="text.secondary"
                >
                  Total Entries
                </Typography>
              </Box>
            </StatItem>
          </motion.div>
        </Grid>
        
        <Grid item xs={6}>
          <motion.div variants={itemVariants}>
            <StatItem>
              <IconContainer>
                <TrendingUpIcon color="success" fontSize="small" />
              </IconContainer>
              <Box>
                <Typography variant="body2" fontWeight={500}>
                  Growth Rate
                </Typography>
                <Typography 
                  variant="h6" 
                  color="success.main" 
                  sx={{ 
                    fontWeight: 500,
                    fontSize: '1.1rem',
                    lineHeight: 1.2,
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  {knowledgeStats.growthRate}%
                  <TrendingUpIcon 
                    fontSize="small" 
                    sx={{ ml: 0.5, fontSize: '1rem' }}
                  />
                </Typography>
                <Typography 
                  variant="caption" 
                  color="text.secondary"
                >
                  Last 30 days
                </Typography>
              </Box>
            </StatItem>
          </motion.div>
        </Grid>
      </Grid>
      
      <motion.div variants={itemVariants}>
        <Box mb={2}>
          <ProgressLabel>
            <Typography variant="body2">Knowledge Freshness</Typography>
            <Typography variant="body2" color="primary">
              {knowledgeStats.freshness}%
            </Typography>
          </ProgressLabel>
          <Tooltip title="Percentage of knowledge updated within the last 3 months">
            <LinearProgress
              variant="determinate"
              value={knowledgeStats.freshness}
              color="secondary"
              sx={{
                height: 6,
                borderRadius: 3,
                '& .MuiLinearProgress-bar': {
                  borderRadius: 3,
                },
              }}
            />
          </Tooltip>
        </Box>
      </motion.div>
      
      <motion.div variants={itemVariants}>
        <Box mb={3}>
          <ProgressLabel>
            <Typography variant="body2">Confidence Score</Typography>
            <Typography variant="body2" color="primary">
              {knowledgeStats.confidenceScore}%
            </Typography>
          </ProgressLabel>
          <Tooltip title="Overall confidence level in knowledge accuracy">
            <LinearProgress
              variant="determinate"
              value={knowledgeStats.confidenceScore}
              color="primary"
              sx={{
                height: 6,
                borderRadius: 3,
                '& .MuiLinearProgress-bar': {
                  borderRadius: 3,
                },
              }}
            />
          </Tooltip>
        </Box>
      </motion.div>
      
      <motion.div variants={itemVariants}>
        <Typography 
          variant="body2" 
          fontWeight={500} 
          sx={{ mb: 1.5, display: 'flex', alignItems: 'center' }}
        >
          <PsychologyIcon fontSize="small" sx={{ mr: 0.75 }} color="primary" />
          Top Knowledge Domains
        </Typography>
        
        {knowledgeStats.topTopics.map((topic, index) => (
          <motion.div 
            key={topic.name}
            variants={topicVariants}
            custom={index}
          >
            <TopicContainer>
              <Typography variant="body2">{topic.name}</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ mr: 1 }}
                >
                  {formatNumber(topic.entries)}
                </Typography>
                <Box
                  sx={{
                    width: 36,
                    mr: -0.5,
                  }}
                >
                  <LinearProgress
                    variant="determinate"
                    value={(topic.entries / totalTopicEntries) * 100}
                    sx={{
                      height: 4,
                      borderRadius: 2,
                      bgcolor: 'rgba(0, 0, 0, 0.05)',
                    }}
                  />
                </Box>
              </Box>
            </TopicContainer>
          </motion.div>
        ))}
      </motion.div>
      
      <Box 
        sx={{ 
          textAlign: 'center', 
          mt: 1.5,
          color: 'text.secondary',
          fontSize: '0.75rem',
          opacity: 0.7
        }}
        component={motion.div}
        variants={itemVariants}
      >
        <NewReleasesIcon 
          fontSize="small" 
          sx={{ 
            verticalAlign: 'middle', 
            mr: 0.5,
            fontSize: '0.875rem'
          }} 
        />
        15 new entries added today
      </Box>
    </StatsContainer>
  );
};

export default KnowledgeStats;
